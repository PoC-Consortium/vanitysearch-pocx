# Rust: CUDA Backend

## src/gpu/cuda/mod.rs

```rust
mod context;
pub use context::CudaBackend;
```

## src/gpu/cuda/context.rs

```rust
use crate::gpu::GpuBackend;
use crate::result::GpuFoundItem;
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
        
        let mut backend = Self {
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
        
        backend.upload_generator_table()?;
        
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
    
    fn upload_generator_table(&mut self) -> Result<()> {
        use crate::crypto::secp256k1::constants::*;
        
        // Compute Gn[i] = (i+1)*G for i = 0..GRP_SIZE/2-1
        let secp = secp256k1::Secp256k1::new();
        let g = secp256k1::PublicKey::from_slice(&[
            0x04, // uncompressed
            // Gx
            0x79, 0xBE, 0x66, 0x7E, 0xF9, 0xDC, 0xBB, 0xAC,
            0x55, 0xA0, 0x62, 0x95, 0xCE, 0x87, 0x0B, 0x07,
            0x02, 0x9B, 0xFC, 0xDB, 0x2D, 0xCE, 0x28, 0xD9,
            0x59, 0xF2, 0x81, 0x5B, 0x16, 0xF8, 0x17, 0x98,
            // Gy
            0x48, 0x3A, 0xDA, 0x77, 0x26, 0xA3, 0xC4, 0x65,
            0x5D, 0xA4, 0xFB, 0xFC, 0x0E, 0x11, 0x08, 0xA8,
            0xFD, 0x17, 0xB4, 0x48, 0xA6, 0x85, 0x54, 0x19,
            0x9C, 0x47, 0xD0, 0x8F, 0xFB, 0x10, 0xD4, 0xB8,
        ])?;
        
        // For simplicity, use pre-computed constants from C++ version
        // In production, compute dynamically
        
        self.device.htod_copy_into(GX.to_vec(), &mut self.gn_x)?;
        self.device.htod_copy_into(GY.to_vec(), &mut self.gn_y)?;
        
        Ok(())
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
            if let Some(data) = crate::crypto::bech32::decode_prefix(prefix) {
                mark_prefix_entries(&mut table, &data);
            }
        }
        
        self.device.htod_copy_into(table, &mut self.prefix_table)?;
        Ok(())
    }
    
    fn set_keys(&mut self, keys: &[[u8; 32]]) -> Result<()> {
        self.keys = keys.to_vec();
        
        // Compute starting points and upload
        let crypto = crate::crypto::Crypto::new();
        let mut x_data = vec![0u64; self.num_threads * 4];
        let mut y_data = vec![0u64; self.num_threads * 4];
        
        for (i, key) in keys.iter().enumerate() {
            // Compute (key + GRP_SIZE/2) * G
            // Simplified: just use the key directly for now
            let pubkey = crypto.public_key(key)?;
            // Extract x, y and convert to u64 arrays
            // ...
        }
        
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
        // In full implementation, update self.keys
    }
}

fn mark_prefix_entries(table: &mut [u16], prefix_data: &[u8]) {
    let bits = prefix_data.len() * 5;
    
    if bits >= 16 {
        let prefix16 = (prefix_data[0] as u16) << 11 
                     | (prefix_data[1] as u16) << 6
                     | (prefix_data[2] as u16) << 1;
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
```

## src/gpu/cuda/kernels/vanity.cu

```cuda
// Include the original CUDA kernels from VanitySearch
// Structure matches the C++ implementation

#include <stdint.h>

#define GRP_SIZE 128
#define MAX_FOUND 64

// ... (Full CUDA kernel implementation)
// See original GPUCompute.cu for reference

extern "C" __global__ void vanity_search(
    uint64_t* startX,
    uint64_t* startY,
    uint64_t* GnX,
    uint64_t* GnY,
    uint64_t* _2GnX,
    uint64_t* _2GnY,
    uint16_t* prefix,
    uint32_t* output,
    uint32_t maxFound
) {
    // Kernel implementation
    // ...
}
```
