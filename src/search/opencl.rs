//! OpenCL-based vanity search engine
//! Provides GPU acceleration on AMD, Intel, and NVIDIA OpenCL-compatible devices

#[cfg(feature = "opencl")]
mod opencl_impl {
    use crate::pattern::Pattern;
    use crate::search::Match;
    use crate::secp256k1::{Point, Scalar, G};
    use ocl::{Buffer, Context, Device, Kernel, Platform, Program, Queue};
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::sync::{Arc, Mutex};

    // Constants matching VanitySearch
    const STEP_SIZE: usize = 16384;
    const ITEM_SIZE32: usize = 8;
    const GRP_SIZE: usize = 1024;

    /// OpenCL search engine configuration
    #[derive(Debug, Clone)]
    pub struct OpenClSearchConfig {
        pub platform_id: usize,
        pub device_id: usize,
        pub pattern: Pattern,
        pub hrp: String,
        pub max_matches: u64,
        pub timeout_secs: u64,
        pub num_threads: usize,
    }

    /// OpenCL search engine
    pub struct OpenClSearchEngine {
        config: OpenClSearchConfig,
        platform: Platform,
        device: Device,
        #[allow(dead_code)]
        context: Context,
        queue: Queue,
        #[allow(dead_code)]
        program: Program,
        kernel: Kernel,
        // Device buffers
        startx_buffer: Buffer<u64>,
        starty_buffer: Buffer<u64>,
        pattern_buffer: Buffer<u8>,
        output_buffer: Buffer<u32>,
        // State
        stop: Arc<AtomicBool>,
        keys_checked: Arc<AtomicU64>,
        matches_found: Arc<AtomicU64>,
        keys_per_launch: u64,
        pattern_5bit: Vec<u8>,
        max_found: usize,
        mutable_state: Mutex<OpenClMutableState>,
    }

    struct OpenClMutableState {
        base_keys: Vec<Scalar>,
        batches_since_regen: u64,
    }

    impl OpenClSearchEngine {
        /// List available OpenCL platforms
        pub fn list_platforms() -> Vec<String> {
            Platform::list()
                .iter()
                .map(|p| p.name().unwrap_or_else(|_| "Unknown".to_string()))
                .collect()
        }

        /// List available OpenCL devices for a platform
        pub fn list_devices(platform_idx: usize) -> Vec<String> {
            let platforms = Platform::list();
            if platform_idx >= platforms.len() {
                return Vec::new();
            }

            Device::list_all(&platforms[platform_idx])
                .map(|devices| {
                    devices
                        .iter()
                        .map(|d| d.name().unwrap_or_else(|_| "Unknown".to_string()))
                        .collect()
                })
                .unwrap_or_default()
        }

        /// Create a new OpenCL search engine
        pub fn new(config: OpenClSearchConfig) -> Result<Self, String> {
            // Get platform
            let platforms = Platform::list();
            if config.platform_id >= platforms.len() {
                return Err(format!(
                    "Platform {} not found ({} available)",
                    config.platform_id,
                    platforms.len()
                ));
            }
            let platform = platforms[config.platform_id];

            // Get device
            let devices = Device::list_all(&platform)
                .map_err(|e| format!("Failed to list devices: {}", e))?;
            if config.device_id >= devices.len() {
                return Err(format!(
                    "Device {} not found ({} available)",
                    config.device_id,
                    devices.len()
                ));
            }
            let device = devices[config.device_id];

            // Create context
            let context = Context::builder()
                .platform(platform)
                .devices(device)
                .build()
                .map_err(|e| format!("Failed to create context: {}", e))?;

            // Create command queue
            let queue = Queue::new(&context, device, None)
                .map_err(|e| format!("Failed to create queue: {}", e))?;

            // Load kernel source files
            let group_source = include_str!("../../opencl/group.cl");
            let math_source = include_str!("../../opencl/math.cl");
            let hash_source = include_str!("../../opencl/hash.cl");
            let kernel_source = include_str!("../../opencl/bech32_kernel.cl");

            // Combine sources (order matters)
            let combined_source = format!(
                "{}\n{}\n{}\n{}",
                group_source, math_source, hash_source, kernel_source
            );

            // Build program
            let program = Program::builder()
                .src(combined_source)
                .devices(device)
                .cmplr_opt(format!("-DGRP_SIZE={} -DSTEP_SIZE={}", GRP_SIZE, STEP_SIZE))
                .build(&context)
                .map_err(|e| format!("Failed to build program: {}", e))?;

            // Create kernel
            let kernel = Kernel::builder()
                .program(&program)
                .name("bech32_search")
                .queue(queue.clone())
                .arg(None::<&Buffer<u64>>) // startx
                .arg(None::<&Buffer<u64>>) // starty
                .arg(None::<&Buffer<u8>>) // pattern
                .arg(0i32) // pattern_len
                .arg(0u32) // max_found
                .arg(None::<&Buffer<u32>>) // out
                .build()
                .map_err(|e| format!("Failed to create kernel: {}", e))?;

            let max_found = 65536usize;
            let num_threads = config.num_threads;

            // Allocate buffers
            // startx/starty: 4 x uint64 per thread (SoA layout)
            let coord_size = num_threads * 4;

            let startx_buffer = Buffer::<u64>::builder()
                .queue(queue.clone())
                .len(coord_size)
                .build()
                .map_err(|e| format!("Failed to create startx buffer: {}", e))?;

            let starty_buffer = Buffer::<u64>::builder()
                .queue(queue.clone())
                .len(coord_size)
                .build()
                .map_err(|e| format!("Failed to create starty buffer: {}", e))?;

            // Pattern buffer: max 32 bytes
            let pattern_buffer = Buffer::<u8>::builder()
                .queue(queue.clone())
                .len(32)
                .build()
                .map_err(|e| format!("Failed to create pattern buffer: {}", e))?;

            // Output buffer: [count] + max_found * ITEM_SIZE32
            let output_size = 1 + max_found * ITEM_SIZE32;
            let output_buffer = Buffer::<u32>::builder()
                .queue(queue.clone())
                .len(output_size)
                .build()
                .map_err(|e| format!("Failed to create output buffer: {}", e))?;

            // Convert pattern to 5-bit values
            // data_pattern includes witness version 'q' at the start - skip it
            // For "bc1qtest", data_pattern is "qtest", we want "test"
            let pattern_without_witness = if config.pattern.data_pattern.starts_with('q') {
                &config.pattern.data_pattern[1..]
            } else {
                &config.pattern.data_pattern
            };
            let pattern_5bit = Self::pattern_to_5bit(pattern_without_witness)?;

            let keys_per_launch = (num_threads * STEP_SIZE * 6) as u64;

            Ok(Self {
                config,
                platform,
                device,
                context,
                queue,
                program,
                kernel,
                startx_buffer,
                starty_buffer,
                pattern_buffer,
                output_buffer,
                stop: Arc::new(AtomicBool::new(false)),
                keys_checked: Arc::new(AtomicU64::new(0)),
                matches_found: Arc::new(AtomicU64::new(0)),
                keys_per_launch,
                pattern_5bit,
                max_found,
                mutable_state: Mutex::new(OpenClMutableState {
                    base_keys: Vec::new(),
                    batches_since_regen: 0,
                }),
            })
        }

        /// Convert bech32 pattern to 5-bit values
        fn pattern_to_5bit(pattern: &str) -> Result<Vec<u8>, String> {
            const BECH32_REV: [i8; 128] = [
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, 15, -1, 10, 17, 21, 20, 26, 30, 7, 5, -1, -1, -1, -1, -1,
                -1, -1, 29, -1, 24, 13, 25, 9, 8, 23, -1, 18, 22, 31, 27, 19, -1, 1, 0, 3, 16, 11,
                28, 12, 14, 6, 4, 2, -1, -1, -1, -1, -1, -1, 29, -1, 24, 13, 25, 9, 8, 23, -1, 18,
                22, 31, 27, 19, -1, 1, 0, 3, 16, 11, 28, 12, 14, 6, 4, 2, -1, -1, -1, -1, -1,
            ];

            let mut result = Vec::with_capacity(pattern.len());
            for c in pattern.chars() {
                let idx = c as usize;
                if idx >= 128 {
                    return Err(format!("Invalid bech32 character: {}", c));
                }
                let val = BECH32_REV[idx];
                if val < 0 {
                    return Err(format!("Invalid bech32 character: {}", c));
                }
                result.push(val as u8);
            }
            Ok(result)
        }

        /// Initialize random starting keys
        fn init_random_keys(&self) -> Result<Vec<Point>, String> {
            use rand::Rng;
            use rayon::prelude::*;

            let mut rng = rand::thread_rng();
            let num_threads = self.config.num_threads;

            const BATCH_SIZE: usize = 256;
            let num_batches = num_threads.div_ceil(BATCH_SIZE);

            let thread_spacing = (num_threads as u64) * (STEP_SIZE as u64);
            let spacing_scalar = Scalar::from_u64(thread_spacing);
            let spacing_g = G.mul(&spacing_scalar);

            let offset_points: Vec<Point> = {
                let mut table = Vec::with_capacity(BATCH_SIZE);
                let mut p = Point::INFINITY;
                for _ in 0..BATCH_SIZE {
                    table.push(p);
                    p = p.add(&spacing_g);
                }
                table
            };

            let batch_scalars: Vec<Scalar> = (0..num_batches)
                .map(|_| {
                    let mut bytes = [0u8; 32];
                    rng.fill(&mut bytes);
                    bytes[0] &= 0x7F;
                    Scalar::from_bytes(&bytes)
                })
                .collect();

            let batch_points: Vec<Point> = batch_scalars.par_iter().map(|key| G.mul(key)).collect();

            let scalars: Vec<Scalar> = (0..num_threads)
                .map(|i| {
                    let batch_idx = i / BATCH_SIZE;
                    let offset = i % BATCH_SIZE;
                    let offset_scalar = spacing_scalar.mul(&Scalar::from_u64(offset as u64));
                    batch_scalars[batch_idx].add(&offset_scalar)
                })
                .collect();

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

            let mut state = self.mutable_state.lock().unwrap();
            state.base_keys = scalars;
            state.batches_since_regen = 0;

            Ok(points)
        }

        /// Copy points to device (SoA layout)
        fn copy_points_to_device(&self, points: &[Point]) -> Result<(), String> {
            let num_threads = self.config.num_threads;
            let mut startx = vec![0u64; num_threads * 4];
            let mut starty = vec![0u64; num_threads * 4];

            for (i, point) in points.iter().enumerate() {
                for j in 0..4 {
                    startx[i + j * num_threads] = point.x.d[j];
                    starty[i + j * num_threads] = point.y.d[j];
                }
            }

            self.startx_buffer
                .write(&startx)
                .enq()
                .map_err(|e| format!("Failed to write startx: {}", e))?;

            self.starty_buffer
                .write(&starty)
                .enq()
                .map_err(|e| format!("Failed to write starty: {}", e))?;

            Ok(())
        }

        /// Set pattern on device
        fn set_pattern(&self) -> Result<(), String> {
            let mut pattern_padded = vec![0u8; 32];
            let len = self.pattern_5bit.len().min(32);
            pattern_padded[..len].copy_from_slice(&self.pattern_5bit[..len]);

            self.pattern_buffer
                .write(&pattern_padded)
                .enq()
                .map_err(|e| format!("Failed to write pattern: {}", e))?;

            Ok(())
        }

        /// Clear output buffer
        fn clear_output(&self) -> Result<(), String> {
            let zeros = vec![0u32; 1 + self.max_found * ITEM_SIZE32];
            self.output_buffer
                .write(&zeros)
                .enq()
                .map_err(|e| format!("Failed to clear output: {}", e))?;
            // Ensure the write completes before kernel launch
            self.queue
                .finish()
                .map_err(|e| format!("Failed to finish after clear: {}", e))?;
            Ok(())
        }

        /// Launch kernel
        fn launch_kernel(&self) -> Result<(), String> {
            let pattern_len = self.pattern_5bit.len() as i32;
            let max_found = self.max_found as u32;

            unsafe {
                self.kernel
                    .set_arg(0, &self.startx_buffer)
                    .map_err(|e| format!("Failed to set arg 0: {}", e))?;
                self.kernel
                    .set_arg(1, &self.starty_buffer)
                    .map_err(|e| format!("Failed to set arg 1: {}", e))?;
                self.kernel
                    .set_arg(2, &self.pattern_buffer)
                    .map_err(|e| format!("Failed to set arg 2: {}", e))?;
                self.kernel
                    .set_arg(3, &pattern_len)
                    .map_err(|e| format!("Failed to set arg 3: {}", e))?;
                self.kernel
                    .set_arg(4, &max_found)
                    .map_err(|e| format!("Failed to set arg 4: {}", e))?;
                self.kernel
                    .set_arg(5, &self.output_buffer)
                    .map_err(|e| format!("Failed to set arg 5: {}", e))?;

                self.kernel
                    .cmd()
                    .global_work_size(self.config.num_threads)
                    .local_work_size(256)
                    .enq()
                    .map_err(|e| format!("Kernel launch failed: {}", e))?;
            }

            Ok(())
        }

        /// Wait for kernel completion
        fn synchronize(&self) -> Result<(), String> {
            self.queue
                .finish()
                .map_err(|e| format!("Queue finish failed: {}", e))?;
            Ok(())
        }

        /// Get results from device
        fn get_results(&self) -> Result<(u32, Vec<u32>), String> {
            let output_size = 1 + self.max_found * ITEM_SIZE32;
            let mut output = vec![0u32; output_size];

            self.output_buffer
                .read(&mut output)
                .enq()
                .map_err(|e| format!("Failed to read output: {}", e))?;

            let match_count = output[0];
            Ok((match_count, output))
        }

        /// Update base keys after kernel run
        fn update_base_keys(&self) {
            let step_scalar = Scalar::from_u64(STEP_SIZE as u64);
            let mut state = self.mutable_state.lock().unwrap();

            for key in state.base_keys.iter_mut() {
                *key = key.add(&step_scalar);
            }
            state.batches_since_regen += 1;
        }

        /// Check if we need to regenerate keys
        fn needs_key_regen(&self) -> bool {
            let num_threads = self.config.num_threads as u64;
            let state = self.mutable_state.lock().unwrap();
            state.batches_since_regen >= (num_threads * 9 / 10)
        }

        /// Process match results
        fn process_matches(
            &self,
            results: &[u32],
            match_count: u32,
            results_tx: &std::sync::mpsc::Sender<Match>,
        ) {
            let state = self.mutable_state.lock().unwrap();

            for i in 0..match_count as usize {
                let base = 1 + i * ITEM_SIZE32;
                if base + ITEM_SIZE32 > results.len() {
                    break;
                }

                let tid = results[base] as usize;
                let info = results[base + 1];
                let incr = ((info >> 16) as i16) as i32;
                let endo = (info & 0x7) as i32;

                let mut hash160 = [0u8; 20];
                for j in 0..5 {
                    let val = results[base + 2 + j];
                    hash160[j * 4] = (val & 0xFF) as u8;
                    hash160[j * 4 + 1] = ((val >> 8) & 0xFF) as u8;
                    hash160[j * 4 + 2] = ((val >> 16) & 0xFF) as u8;
                    hash160[j * 4 + 3] = ((val >> 24) & 0xFF) as u8;
                }

                const GRP_SIZE_HALF: i32 = (GRP_SIZE / 2) as i32;

                if tid < state.base_keys.len() {
                    let base_key = state.base_keys[tid];
                    let incr_abs = if incr >= 0 { incr } else { -incr };
                    let offset = incr_abs - GRP_SIZE_HALF;

                    // Compute the scalar for the point: base_key + offset
                    let mut private_key = base_key;
                    if offset >= 0 {
                        private_key = private_key.add(&Scalar::from_u64(offset as u64));
                    } else {
                        private_key = private_key.sub(&Scalar::from_u64((-offset) as u64));
                    }

                    // Apply endomorphism scalar multiplication BEFORE y-parity check
                    // (following CUDA's approach)
                    if endo == 1 {
                        let lambda1_bytes: [u8; 32] = [
                            0x53, 0x63, 0xad, 0x4c, 0xc0, 0x5c, 0x30, 0xe0, 0xa5, 0x26, 0x1c, 0x02,
                            0x88, 0x12, 0x64, 0x5a, 0x12, 0x2e, 0x22, 0xea, 0x20, 0x81, 0x66, 0x78,
                            0xdf, 0x02, 0x96, 0x7c, 0x1b, 0x23, 0xbd, 0x72,
                        ];
                        let lambda1 = Scalar::from_bytes(&lambda1_bytes);
                        private_key = private_key.mul(&lambda1);
                    } else if endo == 2 {
                        let lambda2_bytes: [u8; 32] = [
                            0xac, 0x9c, 0x52, 0xb3, 0x3f, 0xa3, 0xcf, 0x1f, 0x5a, 0xd9, 0xe3, 0xfd,
                            0x77, 0xed, 0x9b, 0xa4, 0xa8, 0x80, 0xb9, 0xfc, 0x8e, 0xc7, 0x39, 0xc2,
                            0xe0, 0xcf, 0xc8, 0x10, 0xb5, 0x12, 0x83, 0xce,
                        ];
                        let lambda2 = Scalar::from_bytes(&lambda2_bytes);
                        private_key = private_key.mul(&lambda2);
                    }

                    // Check y-parity and negate if needed (following CUDA's approach)
                    // incr < 0 means the GPU matched h2 (03 prefix, odd y)
                    // incr >= 0 means the GPU matched h1 (02 prefix, even y)
                    let gpu_wants_odd_y = incr < 0;
                    let pubpoint = G.mul(&private_key);
                    let scalar_has_odd_y = pubpoint.y.d[0] & 1 != 0;

                    if gpu_wants_odd_y != scalar_has_odd_y {
                        private_key = private_key.neg();
                    }

                    // Compute address from reconstructed private key
                    let final_pubpoint = G.mul(&private_key);
                    let pubkey_bytes = final_pubpoint.to_compressed();
                    let computed_hash160 = crate::hash::hash160_33(&pubkey_bytes);
                    let computed_address =
                        crate::bech32::address_from_hash160(&self.config.hrp, &computed_hash160);

                    // Verify the computed address matches the kernel's hash160
                    let kernel_address =
                        crate::bech32::address_from_hash160(&self.config.hrp, &hash160);
                    if kernel_address != computed_address {
                        // Skip mismatched results (should not happen in production)
                        continue;
                    }

                    // Send verified match using the computed address (which matches the private key)
                    let m = Match {
                        address: computed_address,
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

            // Set pattern
            if let Err(e) = self.set_pattern() {
                eprintln!("[OpenCL] Error setting pattern: {}", e);
                return;
            }

            // Initialize random starting keys
            let points = match self.init_random_keys() {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("[OpenCL] Error initializing keys: {}", e);
                    return;
                }
            };

            // Copy initial keys to device
            if let Err(e) = self.copy_points_to_device(&points) {
                eprintln!("[OpenCL] Error copying initial keys: {}", e);
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
                if let Err(e) = self.clear_output() {
                    eprintln!("[OpenCL] Failed to clear output: {}", e);
                    break;
                }

                // Launch kernel
                if let Err(e) = self.launch_kernel() {
                    eprintln!("[OpenCL] Kernel launch failed: {}", e);
                    break;
                }

                // Synchronize
                if let Err(e) = self.synchronize() {
                    eprintln!("[OpenCL] Sync failed: {}", e);
                    break;
                }

                // Get results
                let (match_count, results) = match self.get_results() {
                    Ok(r) => r,
                    Err(e) => {
                        eprintln!("[OpenCL] Failed to get results: {}", e);
                        break;
                    }
                };

                // Process matches
                if match_count > 0 {
                    let valid_count = match_count.min(self.max_found as u32);
                    self.process_matches(&results, valid_count, &results_tx);
                }

                // Update stats
                self.keys_checked
                    .fetch_add(self.keys_per_launch, Ordering::Relaxed);

                // Update base keys
                self.update_base_keys();

                // Regenerate keys if needed
                if self.needs_key_regen() {
                    let points = match self.init_random_keys() {
                        Ok(p) => p,
                        Err(e) => {
                            eprintln!("[OpenCL] Error regenerating keys: {}", e);
                            break;
                        }
                    };
                    if let Err(e) = self.copy_points_to_device(&points) {
                        eprintln!("[OpenCL] Error copying new keys: {}", e);
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

        pub fn device_name(&self) -> String {
            self.device.name().unwrap_or_else(|_| "Unknown".to_string())
        }

        pub fn platform_name(&self) -> String {
            self.platform
                .name()
                .unwrap_or_else(|_| "Unknown".to_string())
        }
    }

    // Safety: OpenClSearchEngine uses thread-safe types
    unsafe impl Send for OpenClSearchEngine {}
    unsafe impl Sync for OpenClSearchEngine {}
}

#[cfg(feature = "opencl")]
pub use opencl_impl::*;

// Stub implementation when OpenCL is not enabled
#[cfg(not(feature = "opencl"))]
mod stub_impl {
    use crate::pattern::Pattern;
    use crate::search::Match;

    #[derive(Debug, Clone)]
    pub struct OpenClSearchConfig {
        pub platform_id: usize,
        pub device_id: usize,
        pub pattern: Pattern,
        pub hrp: String,
        pub max_matches: u64,
        pub timeout_secs: u64,
        pub num_threads: usize,
    }

    pub struct OpenClSearchEngine;

    impl OpenClSearchEngine {
        pub fn new(_config: OpenClSearchConfig) -> Result<Self, String> {
            Err("OpenCL support not compiled. Rebuild with --features opencl".to_string())
        }

        pub fn list_platforms() -> Vec<String> {
            Vec::new()
        }

        pub fn list_devices(_platform_idx: usize) -> Vec<String> {
            Vec::new()
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
        pub fn device_name(&self) -> String {
            String::new()
        }
        pub fn platform_name(&self) -> String {
            String::new()
        }
    }
}

#[cfg(not(feature = "opencl"))]
pub use stub_impl::*;
