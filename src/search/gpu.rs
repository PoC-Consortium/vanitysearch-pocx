//! GPU-based vanity search engine using CUDA
//! Based on VanitySearch kernel - uses GPU for EC point operations and hash160

#[cfg(feature = "cuda")]
mod cuda_impl {
    use crate::pattern::Pattern;
    use crate::search::Match;
    use crate::secp256k1::{G, Point, Scalar};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

    // Constants matching VanitySearch
    const STEP_SIZE: usize = 16384; // Keys per thread per kernel launch
    const ITEM_SIZE32: usize = 8; // Match result size in u32s

    // FFI functions from GPUBech32.cu
    #[repr(C)]
    pub struct GpuContext {
        _private: [u8; 0],
    }

    unsafe extern "C" {
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
        batches_since_regen: u64, // Track batches for sequential walk optimization
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
            // For bech32, full address is: bc1q<data><checksum>
            // bc1 = human readable part (hrp)
            // q = witness version (0 in bech32m = 'q')
            // The pattern after "bc1q" is the data pattern to match
            let full_prefix = format!("{}1q", config.hrp);
            let pattern_str =
                if let Some(stripped) = config.pattern.pattern.strip_prefix(&full_prefix) {
                    stripped.to_string()
                } else {
                    config.pattern.data_pattern.clone()
                };
            // Note: No additional stripping needed - "bc1q" already includes witness version

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
                    batches_since_regen: 0,
                }),
            })
        }

        /// Generate random starting keys and compute initial points
        /// Threads are spaced far apart (num_threads * STEP_SIZE) to allow many sequential
        /// kernel batches before needing to regenerate random keys.
        fn init_random_keys(&self) -> Result<(), String> {
            use rand::Rng;
            use rayon::prelude::*;

            let mut rng = rand::thread_rng();
            let num_threads =
                (self.config.num_thread_groups * self.config.threads_per_group) as usize;

            // Space threads by num_threads * STEP_SIZE so we can run many batches sequentially
            // Before overlap occurs. After each kernel, GPU advances each thread by STEP_SIZE.
            // With wide spacing, we can run num_threads batches before overlap!
            const BATCH_SIZE: usize = 256;
            let num_batches = num_threads.div_ceil(BATCH_SIZE);

            // Thread spacing: num_threads * STEP_SIZE
            // This allows sequential walks without overlap for many batches
            let thread_spacing = (num_threads as u64) * (STEP_SIZE as u64);
            let spacing_scalar = Scalar::from_u64(thread_spacing);
            let spacing_g = G.mul(&spacing_scalar);

            // Precompute offset table: 0*spacing*G, 1*spacing*G, ..., 255*spacing*G
            let offset_points: Vec<Point> = {
                let mut table = Vec::with_capacity(BATCH_SIZE);
                let mut p = Point::INFINITY;
                for _ in 0..BATCH_SIZE {
                    table.push(p);
                    p = p.add(&spacing_g);
                }
                table
            };

            // Generate one random scalar per batch
            let batch_scalars: Vec<Scalar> = (0..num_batches)
                .map(|_| {
                    let mut bytes = [0u8; 32];
                    rng.fill(&mut bytes);
                    bytes[0] &= 0x7F; // Ensure valid scalar
                    Scalar::from_bytes(&bytes)
                })
                .collect();

            // Compute base points for each batch in parallel (expensive scalar mult)
            let batch_points: Vec<Point> = batch_scalars.par_iter().map(|key| G.mul(key)).collect();

            // Derive all keys - each thread is spaced by num_threads * STEP_SIZE
            let scalars: Vec<Scalar> = (0..num_threads)
                .map(|i| {
                    let batch_idx = i / BATCH_SIZE;
                    let offset = i % BATCH_SIZE;
                    let offset_scalar = spacing_scalar.mul(&Scalar::from_u64(offset as u64));
                    batch_scalars[batch_idx].add(&offset_scalar)
                })
                .collect();

            // Derive all points from batch points using addition (much faster than scalar mult)
            let points: Vec<Point> = (0..num_threads)
                .into_par_iter()
                .map(|i| {
                    let batch_idx = i / BATCH_SIZE;
                    let offset = i % BATCH_SIZE;
                    if offset == 0 {
                        batch_points[batch_idx]
                    } else {
                        batch_points[batch_idx].add(&offset_points[offset])
                    }
                })
                .collect();

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

            state.base_keys = scalars.clone();
            state.batches_since_regen = 0;

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
        /// Also increments batch counter for sequential walk tracking
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

            // The GPU kernel advances each thread by STEP_SIZE*G
            // Update base_keys to match
            let step_scalar = Scalar::from_u64(STEP_SIZE as u64);

            for key in state.base_keys.iter_mut() {
                *key = key.add(&step_scalar);
            }

            state.batches_since_regen += 1;

            Ok(())
        }

        /// Check if we need to regenerate keys (approaching overlap)
        fn needs_key_regen(&self) -> bool {
            let num_threads =
                (self.config.num_thread_groups * self.config.threads_per_group) as u64;
            let state = self.mutable_state.lock().unwrap();
            // With wide spacing of num_threads * STEP_SIZE, we can run num_threads batches
            // before overlap. Leave some margin (90%) to be safe.
            state.batches_since_regen >= (num_threads * 9 / 10)
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
                let incr = ((info >> 16) as i16) as i32;
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

                const GRP_SIZE_HALF: i32 = 512;

                if tid < state.base_keys.len() {
                    let base_key = state.base_keys[tid];
                    let incr_abs = if incr >= 0 { incr } else { -incr };
                    let offset = incr_abs - GRP_SIZE_HALF;

                    let mut private_key = base_key;

                    use crate::secp256k1::G;

                    if offset >= 0 {
                        private_key = private_key.add(&Scalar::from_u64(offset as u64));
                    } else {
                        private_key = private_key.sub(&Scalar::from_u64((-offset) as u64));
                    }

                    // Apply endomorphism adjustment
                    if endo == 1 {
                        let lambda1_bytes: [u8; 32] = [
                            0x53, 0x63, 0xad, 0x4c, 0xc0, 0x5c, 0x30, 0xe0, 0xa5, 0x26, 0x1c, 0x02,
                            0x88, 0x12, 0x64, 0x5a, 0x12, 0x2e, 0x22, 0xea, 0x20, 0x81, 0x66, 0x78,
                            0xdf, 0x02, 0x96, 0x7c, 0x1b, 0x23, 0xbd, 0x72,
                        ];
                        private_key = private_key.mul(&Scalar::from_bytes(&lambda1_bytes));
                    } else if endo == 2 {
                        let lambda2_bytes: [u8; 32] = [
                            0xac, 0x9c, 0x52, 0xb3, 0x3f, 0xa3, 0xcf, 0x1f, 0x5a, 0xd9, 0xe3, 0xfd,
                            0x77, 0xed, 0x9b, 0xa4, 0xa8, 0x80, 0xb9, 0xfc, 0x8e, 0xc7, 0x39, 0xc2,
                            0xe0, 0xcf, 0xc8, 0x10, 0xb5, 0x12, 0x83, 0xce,
                        ];
                        private_key = private_key.mul(&Scalar::from_bytes(&lambda2_bytes));
                    }

                    // Check y-parity and negate if needed
                    let gpu_wants_odd_y = incr < 0;
                    let pubpoint = G.mul(&private_key);
                    let scalar_has_odd_y = pubpoint.y.d[0] & 1 != 0;

                    if gpu_wants_odd_y != scalar_has_odd_y {
                        private_key = private_key.neg();
                    }

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
            let start_time = std::time::Instant::now();
            let max_found = 65536i32;

            // Result buffer
            let mut results: Vec<u32> = vec![0u32; (max_found as usize) * ITEM_SIZE32];
            let mut match_count: u32 = 0;
            let mut last_report = std::time::Instant::now();

            // Initialize random starting keys
            if let Err(e) = self.init_random_keys() {
                eprintln!("Error initializing keys: {}", e);
                return;
            }

            // Copy initial keys to GPU
            if let Err(e) = self.copy_keys_to_gpu() {
                eprintln!("Error copying initial keys: {}", e);
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
                    eprintln!("Failed to clear output buffer");
                    break;
                }

                // Launch kernel
                let launch_result = unsafe { cuda_bech32_launch(self.ctx, max_found) };
                if launch_result != 0 {
                    eprintln!("Kernel launch failed with code {}", launch_result);
                    break;
                }

                // Synchronize
                if unsafe { cuda_bech32_sync() } != 0 {
                    eprintln!("CUDA sync failed");
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
                    eprintln!("Failed to get results");
                    break;
                }

                // Process any matches
                if match_count > 0 {
                    self.process_matches(&results, match_count, &results_tx);
                }

                // Update stats
                self.keys_checked
                    .fetch_add(self.keys_per_launch, Ordering::Relaxed);

                // Progress report every second (handled by main loop, but keep minimal here)
                if last_report.elapsed().as_secs() >= 1 {
                    last_report = std::time::Instant::now();
                }

                // Sequential walk: get updated keys from GPU and check if regeneration needed
                if let Err(e) = self.get_keys_from_gpu() {
                    eprintln!("Error getting updated keys: {}", e);
                    break;
                }

                // Only regenerate when approaching overlap (every ~num_threads batches)
                if self.needs_key_regen() {
                    if let Err(e) = self.init_random_keys() {
                        eprintln!("Error regenerating keys: {}", e);
                        break;
                    }
                    if let Err(e) = self.copy_keys_to_gpu() {
                        eprintln!("Error copying new keys to GPU: {}", e);
                        break;
                    }
                }
            }
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
