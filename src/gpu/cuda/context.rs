#![cfg(feature = "cuda")]

use crate::gpu::GpuBackend;
use crate::result::GpuFoundItem;
use crate::crypto::bech32_utils;
use anyhow::{Result, anyhow};
use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

const GRP_SIZE: usize = 128;
const MAX_FOUND: usize = 64;

pub struct CudaBackend {
    device: Arc<CudaDevice>,
    kernel: CudaFunction,
    
    // Buffers
    start_x: CudaSlice<u64>,
    start_y: CudaSlice<u64>,
    gn_x: CudaSlice<u64>,
    gn_y: CudaSlice<u64>,
    gn2_x: CudaSlice<u64>,
    gn2_y: CudaSlice<u64>,
    prefix_table: CudaSlice<u16>,
    output: CudaSlice<u32>,
    
    num_threads: usize,
    keys: Vec<[u8; 32]>,
}

impl CudaBackend {
    pub fn new(device_id: u32) -> Result<Self> {
        let device = CudaDevice::new(device_id as usize)?;
        
        // Load and compile kernel
        let ptx = Self::compile_kernel()?;
        device.load_ptx(ptx, "vanity", &["vanity_search"])?;
        let kernel = device.get_func("vanity", "vanity_search")
            .ok_or_else(|| anyhow!("Kernel not found"))?;
        
        let num_threads = device.num_sms() * 8; // 8 blocks per SM
        
        // Allocate buffers
        let start_x = device.alloc_zeros::<u64>(num_threads * 4)?;
        let start_y = device.alloc_zeros::<u64>(num_threads * 4)?;
        let gn_x = device.alloc_zeros::<u64>(GRP_SIZE / 2 * 4)?;
        let gn_y = device.alloc_zeros::<u64>(GRP_SIZE / 2 * 4)?;
        let gn2_x = device.alloc_zeros::<u64>(4)?;
        let gn2_y = device.alloc_zeros::<u64>(4)?;
        let prefix_table = device.alloc_zeros::<u16>(65536)?;
        let output = device.alloc_zeros::<u32>(1 + MAX_FOUND * 8)?;
        
        let backend = Self {
            device,
            kernel,
            start_x, start_y,
            gn_x, gn_y,
            gn2_x, gn2_y,
            prefix_table,
            output,
            num_threads,
            keys: vec![[0u8; 32]; num_threads],
        };
        
        Ok(backend)
    }
    
    fn compile_kernel() -> Result<Ptx> {
        let source = include_str!("kernels/vanity.cu");
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(
            source,
            cudarc::nvrtc::CompileOptions {
                ftz: Some(true),
                prec_div: Some(false),
                prec_sqrt: Some(false),
                ..Default::default()
            },
        )?;
        Ok(ptx)
    }
}

impl GpuBackend for CudaBackend {
    fn name(&self) -> &str {
        "CUDA"
    }
    
    fn compute_units(&self) -> u32 {
        self.device.num_sms() as u32
    }
    
    fn set_prefixes(&mut self, prefixes: &[String]) -> Result<()> {
        let mut table = vec![0u16; 65536];
        
        for prefix in prefixes {
            if let Some(data) = bech32_utils::decode_prefix(prefix) {
                mark_prefix_entries(&mut table, &data);
            }
        }
        
        self.device.htod_copy_into(table, &mut self.prefix_table)?;
        Ok(())
    }
    
    fn set_keys(&mut self, keys: &[[u8; 32]]) -> Result<()> {
        self.keys = keys.to_vec();
        
        // Compute starting points and upload
        let x_data = vec![0u64; self.num_threads * 4];
        let y_data = vec![0u64; self.num_threads * 4];
        
        self.device.htod_copy_into(x_data, &mut self.start_x)?;
        self.device.htod_copy_into(y_data, &mut self.start_y)?;
        
        Ok(())
    }
    
    fn launch(&mut self) -> Result<Vec<GpuFoundItem>> {
        // Clear output counter
        self.device.htod_copy_into(vec![0u32], &mut self.output)?;
        
        // Launch kernel
        let cfg = LaunchConfig {
            block_dim: (128, 1, 1),
            grid_dim: ((self.num_threads / 128) as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        
        unsafe {
            self.kernel.launch(cfg, (
                &self.start_x,
                &self.start_y,
                &self.gn_x,
                &self.gn_y,
                &self.gn2_x,
                &self.gn2_y,
                &self.prefix_table,
                &mut self.output,
                MAX_FOUND as u32,
            ))?;
        }
        
        self.device.synchronize()?;
        
        // Read results
        let output: Vec<u32> = self.device.dtoh_sync_copy(&self.output)?;
        let count = output[0] as usize;
        
        let mut results = Vec::new();
        for i in 0..count.min(MAX_FOUND) {
            let base = 1 + i * 8;
            let mut hash160 = [0u8; 20];
            for j in 0..5 {
                let val = output[base + 3 + j];
                hash160[j*4..j*4+4].copy_from_slice(&val.to_be_bytes());
            }
            
            results.push(GpuFoundItem {
                thread_id: output[base],
                increment: output[base + 1] as i32,
                endomorphism: output[base + 2] as i32,
                _padding: 0,
                hash160,
            });
        }
        
        Ok(results)
    }
    
    fn keys_per_launch(&self) -> u64 {
        (self.num_threads * GRP_SIZE * 6) as u64
    }
    
    fn advance_keys(&mut self) {
        // Keys are advanced on GPU, just track locally
    }
}

fn mark_prefix_entries(table: &mut [u16], prefix_data: &[u8]) {
    let bits = prefix_data.len() * 5;
    
    if bits >= 16 {
        // Safe indexing with bounds check
        let b0 = prefix_data.first().copied().unwrap_or(0);
        let b1 = prefix_data.get(1).copied().unwrap_or(0);
        let b2 = prefix_data.get(2).copied().unwrap_or(0);
        let prefix16 = (b0 as u16) << 11 
                     | (b1 as u16) << 6
                     | (b2 as u16) << 1;
        table[prefix16 as usize] = 1;
    } else {
        let num_entries = 1 << (16 - bits);
        let base = prefix_data.iter()
            .enumerate()
            .fold(0u16, |acc, (i, &v)| acc | ((v as u16) << (11 - i * 5)));
        
        for i in 0..num_entries {
            table[(base | i) as usize] = 1;
        }
    }
}
