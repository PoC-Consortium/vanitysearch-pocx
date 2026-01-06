//! GPU-based vanity search engine using CUDA
//! Based on VanitySearch kernel - uses GPU for EC point operations and hash160

#[cfg(feature = "cuda")]
mod cuda_impl {
    use crate::pattern::Pattern;
    use crate::search::Match;
    use crate::secp256k1::{FieldElement, Point, Scalar, G};
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::sync::Arc;

    // Constants matching VanitySearch
    const GRP_SIZE: usize = 1024;
    const STEP_SIZE: usize = 8192; // Keys per thread per kernel launch (increased from 4096)
    const ITEM_SIZE32: usize = 8; // Match result size in u32s

    // FFI functions from GPUBech32.cu
    #[repr(C)]
    pub struct GpuContext {
        _private: [u8; 0],
    }

    extern "C" {
        fn cuda_bech32_get_device_count() -> i32;
        fn cuda_bech32_get_device_name(device_id: i32, name: *mut i8, max_len: i32);
        fn cuda_bech32_get_device_sm_count(device_id: i32) -> i32;
        fn cuda_bech32_init(device_id: i32) -> i32;

        fn cuda_bech32_create_context(
            num_thread_groups: i32,
            threads_per_group: i32,
            max_found: i32,
        ) -> *mut GpuContext;

        fn cuda_bech32_destroy_context(ctx: *mut GpuContext);

        fn cuda_bech32_set_pattern(ctx: *mut GpuContext, pattern: *const i8, len: i32) -> i32;

        fn cuda_bech32_copy_keys(ctx: *mut GpuContext, keys: *const u8, size: usize) -> i32;

        fn cuda_bech32_clear_output(ctx: *mut GpuContext) -> i32;

        fn cuda_bech32_launch(ctx: *mut GpuContext, max_found: i32) -> i32;

        fn cuda_bech32_sync() -> i32;

        fn cuda_bech32_get_results(
            ctx: *mut GpuContext,
            match_count: *mut u32,
            results: *mut u32,
            max_results: i32,
        ) -> i32;

        fn cuda_bech32_get_keys(ctx: *mut GpuContext, keys: *mut u8, size: usize) -> i32;

        fn cuda_bech32_keys_per_launch(num_thread_groups: i32, threads_per_group: i32) -> u64;
    }

    /// GPU search engine configuration
    #[derive(Debug, Clone)]
    pub struct GpuSearchConfig {
        pub device_id: i32,
        pub pattern: Pattern,
        pub hrp: String,
        pub max_matches: u64,
        pub timeout_secs: u64,
        /// Number of thread blocks
        pub num_thread_groups: i32,
        /// Threads per block (should be 128 or 256)
        pub threads_per_group: i32,
    }

    use std::sync::Mutex;

    /// GPU search engine
    pub struct GpuSearchEngine {
        config: GpuSearchConfig,
        ctx: *mut GpuContext,
        stop: Arc<AtomicBool>,
        keys_checked: Arc<AtomicU64>,
        matches_found: Arc<AtomicU64>,
        keys_per_launch: u64,
        // Host-side key management - protected by Mutex for interior mutability
        mutable_state: Mutex<GpuMutableState>,
    }

    /// Mutable state for GPU engine - protected by mutex
    struct GpuMutableState {
        base_keys: Vec<Scalar>,
        keys_buffer: Vec<u8>,
    }

    // Safety: GpuSearchEngine is used on a single thread
    // The ctx pointer is only accessed from the thread that owns the engine
    // The Arc<AtomicBool> and Arc<AtomicU64> fields are thread-safe
    unsafe impl Send for GpuSearchEngine {}
    unsafe impl Sync for GpuSearchEngine {}

    impl Drop for GpuSearchEngine {
        fn drop(&mut self) {
            if !self.ctx.is_null() {
                unsafe {
                    cuda_bech32_destroy_context(self.ctx);
                }
            }
        }
    }

    impl GpuSearchEngine {
        /// Get CUDA device count
        pub fn device_count() -> i32 {
            unsafe { cuda_bech32_get_device_count() }
        }

        /// Get device name
        pub fn device_name(device_id: i32) -> String {
            let mut name = vec![0i8; 256];
            unsafe { cuda_bech32_get_device_name(device_id, name.as_mut_ptr(), 256) };
            let c_str = unsafe { std::ffi::CStr::from_ptr(name.as_ptr()) };
            c_str.to_string_lossy().into_owned()
        }

        /// Get device SM (multiprocessor) count
        pub fn device_sm_count(device_id: i32) -> i32 {
            unsafe { cuda_bech32_get_device_sm_count(device_id) }
        }

        /// Create a new GPU search engine
        pub fn new(config: GpuSearchConfig) -> Result<Self, String> {
            // Initialize CUDA device
            let result = unsafe { cuda_bech32_init(config.device_id) };
            if result != 0 {
                return Err(format!(
                    "Failed to initialize CUDA device {}",
                    config.device_id
                ));
            }

            let max_found = 65536i32;

            // Create GPU context
            let ctx = unsafe {
                cuda_bech32_create_context(
                    config.num_thread_groups,
                    config.threads_per_group,
                    max_found,
                )
            };

            if ctx.is_null() {
                return Err("Failed to create GPU context".to_string());
            }

            // Extract pattern (after hrp + "1q")
            let full_prefix = format!("{}1q", config.hrp);
            let pattern_str = if config.pattern.pattern.starts_with(&full_prefix) {
                config.pattern.pattern[full_prefix.len()..].to_string()
            } else {
                config.pattern.data_pattern.clone()
            };

            // Remove witness version 'q' if present at start
            let pattern_str = if pattern_str.starts_with('q') {
                pattern_str[1..].to_string()
            } else {
                pattern_str
            };

            eprintln!("[GPU] Search pattern (after bc1q): '{}'", pattern_str);

            // Set pattern
            let pattern_c = std::ffi::CString::new(pattern_str.as_str()).unwrap();
            let result = unsafe {
                cuda_bech32_set_pattern(ctx, pattern_c.as_ptr(), pattern_str.len() as i32)
            };

            if result != 0 {
                unsafe { cuda_bech32_destroy_context(ctx) };
                return Err(format!("Failed to set pattern (error {})", result));
            }

            let keys_per_launch = unsafe {
                cuda_bech32_keys_per_launch(config.num_thread_groups, config.threads_per_group)
            };

            eprintln!(
                "[GPU] {} thread groups × {} threads × {} step × 6 endo = {} keys/launch",
                config.num_thread_groups, config.threads_per_group, STEP_SIZE, keys_per_launch
            );

            // Calculate buffer size: each thread needs 8 uint64 for (x,y) coordinates
            let total_threads = config.num_thread_groups * config.threads_per_group;
            let keys_buffer_size = total_threads as usize * 8 * 8; // 8 uint64 = 64 bytes per thread

            Ok(Self {
                config,
                ctx,
                stop: Arc::new(AtomicBool::new(false)),
                keys_checked: Arc::new(AtomicU64::new(0)),
                matches_found: Arc::new(AtomicU64::new(0)),
                keys_per_launch,
                mutable_state: Mutex::new(GpuMutableState {
                    base_keys: Vec::new(),
                    keys_buffer: vec![0u8; keys_buffer_size],
                }),
            })
        }

        /// Generate random starting keys and compute initial points
        fn init_random_keys(&self) -> Result<(), String> {
            use rand::Rng;
            use rayon::prelude::*;

            let mut rng = rand::thread_rng();
            let num_threads =
                (self.config.num_thread_groups * self.config.threads_per_group) as usize;

            eprintln!("[GPU] Generating {} random starting points...", num_threads);
            let start = std::time::Instant::now();

            // Generate random starting scalars
            let scalars: Vec<Scalar> = (0..num_threads)
                .map(|i| {
                    let mut bytes = [0u8; 32];
                    rng.fill(&mut bytes);
                    bytes[0] &= 0x7F; // Ensure valid scalar
                    let mut s = Scalar::from_bytes(&bytes);
                    // Add thread index to ensure uniqueness
                    let mut i_bytes = [0u8; 32];
                    i_bytes[31] = (i & 0xFF) as u8;
                    i_bytes[30] = ((i >> 8) & 0xFF) as u8;
                    i_bytes[29] = ((i >> 16) & 0xFF) as u8;
                    i_bytes[28] = ((i >> 24) & 0xFF) as u8;
                    let i_scalar = Scalar::from_bytes(&i_bytes);
                    s = s.add(&i_scalar);
                    s
                })
                .collect();

            // Compute initial points in parallel (scalar multiplication)
            let points: Vec<Point> = scalars.par_iter().map(|key| G.mul(key)).collect();

            // Pack points into buffer for GPU using VanitySearch memory layout (SoA)
            // For memory coalescing, layout is per thread group:
            //   x[0] for all threads, x[1] for all threads, ..., y[3] for all threads
            // This matches Load256A macro which uses: a[IDX], a[IDX+blockDim.x], etc.
            let mut state = self.mutable_state.lock().unwrap();
            let threads_per_group = self.config.threads_per_group as usize;
            let num_groups = self.config.num_thread_groups as usize;

            for group in 0..num_groups {
                // Base offset for this thread group (8 uint64 per thread)
                let group_base = group * threads_per_group * 8 * 8; // 8 uint64 * 8 bytes

                for thread in 0..threads_per_group {
                    let point_idx = group * threads_per_group + thread;
                    let point = &points[point_idx];

                    // Write x coordinates: x[0] at offset 0, x[1] at offset threads_per_group, etc.
                    for j in 0..4 {
                        let offset = group_base + (j * threads_per_group + thread) * 8;
                        state.keys_buffer[offset..offset + 8]
                            .copy_from_slice(&point.x.d[j].to_le_bytes());
                    }

                    // Write y coordinates: y[0] at offset 4*threads_per_group, etc.
                    for j in 0..4 {
                        let offset = group_base + ((4 + j) * threads_per_group + thread) * 8;
                        state.keys_buffer[offset..offset + 8]
                            .copy_from_slice(&point.y.d[j].to_le_bytes());
                    }
                }
            }

            state.base_keys = scalars;

            eprintln!("[GPU] Initial point generation took {:?}", start.elapsed());

            Ok(())
        }

        /// Copy keys to GPU
        fn copy_keys_to_gpu(&self) -> Result<(), String> {
            let state = self.mutable_state.lock().unwrap();
            let result = unsafe {
                cuda_bech32_copy_keys(
                    self.ctx,
                    state.keys_buffer.as_ptr(),
                    state.keys_buffer.len(),
                )
            };

            if result != 0 {
                return Err("Failed to copy keys to GPU".to_string());
            }

            Ok(())
        }

        /// Get updated keys from GPU (after kernel processes them)
        fn get_keys_from_gpu(&self) -> Result<(), String> {
            let mut state = self.mutable_state.lock().unwrap();
            let result = unsafe {
                cuda_bech32_get_keys(
                    self.ctx,
                    state.keys_buffer.as_mut_ptr(),
                    state.keys_buffer.len(),
                )
            };

            if result != 0 {
                return Err("Failed to get keys from GPU".to_string());
            }

            // Update base_keys to match GPU state
            // After kernel: start_point += STEP_SIZE * 6 * G (approximately)
            let step_scalar = {
                let n = (STEP_SIZE * 6) as u64;
                let mut bytes = [0u8; 32];
                bytes[24] = (n >> 56) as u8;
                bytes[25] = (n >> 48) as u8;
                bytes[26] = (n >> 40) as u8;
                bytes[27] = (n >> 32) as u8;
                bytes[28] = (n >> 24) as u8;
                bytes[29] = (n >> 16) as u8;
                bytes[30] = (n >> 8) as u8;
                bytes[31] = n as u8;
                Scalar::from_bytes(&bytes)
            };

            for key in state.base_keys.iter_mut() {
                *key = key.add(&step_scalar);
            }

            Ok(())
        }

        /// Process match results from GPU
        fn process_matches(
            &self,
            results: &[u32],
            match_count: u32,
            results_tx: &std::sync::mpsc::Sender<Match>,
        ) {
            let state = self.mutable_state.lock().unwrap();
            for i in 0..match_count as usize {
                let base = i * ITEM_SIZE32;
                if base + ITEM_SIZE32 > results.len() {
                    break;
                }

                let tid = results[base] as usize;
                let info = results[base + 1];
                let incr = (info >> 16) as i32;
                let endo = (info & 0x7) as i32;

                // Extract hash160 from results
                let mut hash160 = [0u8; 20];
                for j in 0..5 {
                    let val = results[base + 2 + j];
                    hash160[j * 4] = (val & 0xFF) as u8;
                    hash160[j * 4 + 1] = ((val >> 8) & 0xFF) as u8;
                    hash160[j * 4 + 2] = ((val >> 16) & 0xFF) as u8;
                    hash160[j * 4 + 3] = ((val >> 24) & 0xFF) as u8;
                }

                // Compute private key: base_key + incr (±) with endomorphism adjustment
                if tid < state.base_keys.len() {
                    let mut private_key = state.base_keys[tid].clone();

                    // Apply increment
                    if incr >= 0 {
                        let mut incr_bytes = [0u8; 32];
                        let incr_val = incr as u64;
                        incr_bytes[24] = (incr_val >> 56) as u8;
                        incr_bytes[25] = (incr_val >> 48) as u8;
                        incr_bytes[26] = (incr_val >> 40) as u8;
                        incr_bytes[27] = (incr_val >> 32) as u8;
                        incr_bytes[28] = (incr_val >> 24) as u8;
                        incr_bytes[29] = (incr_val >> 16) as u8;
                        incr_bytes[30] = (incr_val >> 8) as u8;
                        incr_bytes[31] = incr_val as u8;
                        private_key = private_key.add(&Scalar::from_bytes(&incr_bytes));
                    } else {
                        let mut incr_bytes = [0u8; 32];
                        let incr_val = (-incr) as u64;
                        incr_bytes[24] = (incr_val >> 56) as u8;
                        incr_bytes[25] = (incr_val >> 48) as u8;
                        incr_bytes[26] = (incr_val >> 40) as u8;
                        incr_bytes[27] = (incr_val >> 32) as u8;
                        incr_bytes[28] = (incr_val >> 24) as u8;
                        incr_bytes[29] = (incr_val >> 16) as u8;
                        incr_bytes[30] = (incr_val >> 8) as u8;
                        incr_bytes[31] = incr_val as u8;
                        private_key = private_key.sub(&Scalar::from_bytes(&incr_bytes));
                    }

                    // Apply endomorphism adjustment
                    // For endo=1: k' = k * lambda1 mod n
                    // For endo=2: k' = k * lambda2 mod n
                    // Where lambda1 and lambda2 are cube roots of unity mod n
                    if endo == 1 {
                        // lambda1 = 0x5363ad4cc05c30e0a5261c028812645a122e22ea20816678df02967c1b23bd72
                        let lambda1_bytes: [u8; 32] = [
                            0x53, 0x63, 0xad, 0x4c, 0xc0, 0x5c, 0x30, 0xe0, 0xa5, 0x26, 0x1c, 0x02,
                            0x88, 0x12, 0x64, 0x5a, 0x12, 0x2e, 0x22, 0xea, 0x20, 0x81, 0x66, 0x78,
                            0xdf, 0x02, 0x96, 0x7c, 0x1b, 0x23, 0xbd, 0x72,
                        ];
                        let lambda1 = Scalar::from_bytes(&lambda1_bytes);
                        private_key = private_key.mul(&lambda1);
                    } else if endo == 2 {
                        // lambda2 = lambda1^2 mod n
                        let lambda2_bytes: [u8; 32] = [
                            0xac, 0x9c, 0x52, 0xb3, 0x3f, 0xa3, 0xcf, 0x1f, 0x5a, 0xd9, 0xe3, 0xfd,
                            0x77, 0xed, 0x9b, 0xa4, 0xa8, 0x80, 0xb9, 0xfc, 0x8e, 0xc7, 0x39, 0xc2,
                            0xe0, 0xcf, 0xc8, 0x10, 0xb5, 0x12, 0x83, 0xcf,
                        ];
                        let lambda2 = Scalar::from_bytes(&lambda2_bytes);
                        private_key = private_key.mul(&lambda2);
                    }

                    // If negative increment was used, we need to negate the key
                    // (because we're using the -y version of the point)
                    if incr < 0 {
                        private_key = private_key.neg();
                    }

                    // Create address from hash160
                    let address = crate::bech32::address_from_hash160(&self.config.hrp, &hash160);

                    let m = Match {
                        address,
                        private_key,
                        compressed: true,
                    };

                    self.matches_found.fetch_add(1, Ordering::Relaxed);
                    let _ = results_tx.send(m);
                }
            }
        }

        /// Run the search
        pub fn run(&self, results_tx: std::sync::mpsc::Sender<Match>) {
            eprintln!(
                "[GPU] Starting search: {} blocks × {} threads",
                self.config.num_thread_groups, self.config.threads_per_group
            );

            let start_time = std::time::Instant::now();
            let max_found = 65536i32;

            // Result buffer
            let mut results: Vec<u32> = vec![0u32; (max_found as usize) * ITEM_SIZE32];
            let mut match_count: u32 = 0;
            let mut batch_count = 0u64;
            let mut last_report = std::time::Instant::now();

            // Initialize random starting keys
            if let Err(e) = self.init_random_keys() {
                eprintln!("[GPU] Error initializing keys: {}", e);
                return;
            }

            // Copy initial keys to GPU
            if let Err(e) = self.copy_keys_to_gpu() {
                eprintln!("[GPU] Error copying initial keys: {}", e);
                return;
            }

            while !self.stop.load(Ordering::Relaxed) {
                // Check timeout
                if self.config.timeout_secs > 0
                    && start_time.elapsed().as_secs() >= self.config.timeout_secs
                {
                    break;
                }

                // Check max matches
                if self.config.max_matches > 0
                    && self.matches_found.load(Ordering::Relaxed) >= self.config.max_matches
                {
                    break;
                }

                // Clear output buffer
                if unsafe { cuda_bech32_clear_output(self.ctx) } != 0 {
                    eprintln!("[GPU] Failed to clear output buffer");
                    break;
                }

                // Launch kernel
                let launch_result = unsafe { cuda_bech32_launch(self.ctx, max_found) };
                if launch_result != 0 {
                    eprintln!("[GPU] Kernel launch failed with code {}", launch_result);
                    break;
                }

                // Synchronize
                if unsafe { cuda_bech32_sync() } != 0 {
                    eprintln!("[GPU] CUDA sync failed");
                    break;
                }

                // Get results
                let get_result = unsafe {
                    cuda_bech32_get_results(
                        self.ctx,
                        &mut match_count,
                        results.as_mut_ptr(),
                        max_found,
                    )
                };

                if get_result != 0 {
                    eprintln!("[GPU] Failed to get results");
                    break;
                }

                // Process any matches
                if match_count > 0 {
                    self.process_matches(&results, match_count, &results_tx);
                }

                // Update stats
                self.keys_checked
                    .fetch_add(self.keys_per_launch, Ordering::Relaxed);
                batch_count += 1;

                // Progress report every second
                if last_report.elapsed().as_secs() >= 1 {
                    let elapsed = start_time.elapsed().as_secs_f64();
                    let keys = self.keys_checked.load(Ordering::Relaxed);
                    let rate = keys as f64 / elapsed / 1_000_000.0;
                    let matches = self.matches_found.load(Ordering::Relaxed);
                    eprintln!(
                        "[{:.1}s] {:.2} Mkey/s | {} keys checked | {} matches",
                        elapsed, rate, keys, matches
                    );
                    last_report = std::time::Instant::now();
                }

                // Get updated keys from GPU (kernel advances the starting points)
                if let Err(e) = self.get_keys_from_gpu() {
                    eprintln!("[GPU] Error getting updated keys: {}", e);
                    break;
                }
            }

            let elapsed = start_time.elapsed().as_secs_f64();
            let keys = self.keys_checked.load(Ordering::Relaxed);
            let rate = keys as f64 / elapsed / 1_000_000.0;
            eprintln!(
                "[GPU] Search completed: {} batches, {:.2} Mkey/s average",
                batch_count, rate
            );
        }

        pub fn stop(&self) {
            self.stop.store(true, Ordering::Relaxed);
        }

        pub fn keys_checked(&self) -> u64 {
            self.keys_checked.load(Ordering::Relaxed)
        }

        pub fn matches_found(&self) -> u64 {
            self.matches_found.load(Ordering::Relaxed)
        }

        pub fn is_stopped(&self) -> bool {
            self.stop.load(Ordering::Relaxed)
        }
    }
}

#[cfg(feature = "cuda")]
pub use cuda_impl::*;

// Stub implementation when CUDA is not enabled
#[cfg(not(feature = "cuda"))]
mod stub_impl {
    use crate::pattern::Pattern;
    use crate::search::Match;

    #[derive(Debug, Clone)]
    pub struct GpuSearchConfig {
        pub device_id: i32,
        pub pattern: Pattern,
        pub hrp: String,
        pub max_matches: u64,
        pub timeout_secs: u64,
        pub num_thread_groups: i32,
        pub threads_per_group: i32,
    }

    pub struct GpuSearchEngine;

    impl GpuSearchEngine {
        pub fn new(_config: GpuSearchConfig) -> Result<Self, String> {
            Err("CUDA support not compiled. Rebuild with --features cuda".to_string())
        }

        pub fn device_count() -> i32 {
            0
        }

        pub fn device_name(_device_id: i32) -> String {
            String::new()
        }

        pub fn run(&self, _results_tx: std::sync::mpsc::Sender<Match>) {}
        pub fn stop(&self) {}
        pub fn keys_checked(&self) -> u64 {
            0
        }
        pub fn matches_found(&self) -> u64 {
            0
        }
        pub fn is_stopped(&self) -> bool {
            true
        }
    }
}

#[cfg(not(feature = "cuda"))]
pub use stub_impl::*;
