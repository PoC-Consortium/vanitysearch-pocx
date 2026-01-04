# Rust: OpenCL Backend

## src/gpu/opencl/mod.rs

```rust
mod context;
pub use context::OpenCLBackend;
```

## src/gpu/opencl/context.rs

```rust
use crate::gpu::GpuBackend;
use crate::result::GpuFoundItem;
use anyhow::{Result, anyhow};
use opencl3::command_queue::CommandQueue;
use opencl3::context::Context;
use opencl3::device::{Device, CL_DEVICE_TYPE_GPU};
use opencl3::kernel::{Kernel, ExecuteKernel};
use opencl3::memory::{Buffer, CL_MEM_READ_WRITE, CL_MEM_READ_ONLY};
use opencl3::platform::get_platforms;
use opencl3::program::Program;
use opencl3::types::{cl_ulong, cl_uint, cl_ushort};
use std::ptr;

const GRP_SIZE: usize = 128;
const MAX_FOUND: usize = 64;

pub struct OpenCLBackend {
    context: Context,
    queue: CommandQueue,
    kernel: Kernel,
    
    // Buffers
    start_x: Buffer<cl_ulong>,
    start_y: Buffer<cl_ulong>,
    gn_x: Buffer<cl_ulong>,
    gn_y: Buffer<cl_ulong>,
    gn2_x: Buffer<cl_ulong>,
    gn2_y: Buffer<cl_ulong>,
    prefix_table: Buffer<cl_ushort>,
    output: Buffer<cl_uint>,
    
    num_threads: usize,
    device_name: String,
    compute_units: u32,
    keys: Vec<[u8; 32]>,
}

impl OpenCLBackend {
    pub fn new(device_id: u32) -> Result<Self> {
        let platforms = get_platforms()?;
        
        // Find GPU device
        let mut gpu_device = None;
        let mut current_id = 0u32;
        
        'outer: for platform in &platforms {
            let devices = platform.get_devices(CL_DEVICE_TYPE_GPU)?;
            for device in devices {
                if current_id == device_id {
                    gpu_device = Some(device);
                    break 'outer;
                }
                current_id += 1;
            }
        }
        
        let device = gpu_device.ok_or_else(|| anyhow!("GPU device {} not found", device_id))?;
        let device_name = device.name()?;
        let compute_units = device.max_compute_units()?;
        
        // Create context and queue
        let context = Context::from_device(&device)?;
        let queue = CommandQueue::create_default(&context, 0)?;
        
        // Compile kernel
        let source = include_str!("kernels/vanity.cl");
        let program = Program::create_and_build_from_source(
            &context,
            source,
            &format!("-D GRP_SIZE={} -D MAX_FOUND={}", GRP_SIZE, MAX_FOUND),
        )?;
        let kernel = Kernel::create(&program, "VanitySearch")?;
        
        let num_threads = (compute_units as usize) * 8;
        
        // Allocate buffers
        let start_x = Buffer::create(&context, CL_MEM_READ_WRITE, num_threads * 4, ptr::null_mut())?;
        let start_y = Buffer::create(&context, CL_MEM_READ_WRITE, num_threads * 4, ptr::null_mut())?;
        let gn_x = Buffer::create(&context, CL_MEM_READ_ONLY, GRP_SIZE / 2 * 4, ptr::null_mut())?;
        let gn_y = Buffer::create(&context, CL_MEM_READ_ONLY, GRP_SIZE / 2 * 4, ptr::null_mut())?;
        let gn2_x = Buffer::create(&context, CL_MEM_READ_ONLY, 4, ptr::null_mut())?;
        let gn2_y = Buffer::create(&context, CL_MEM_READ_ONLY, 4, ptr::null_mut())?;
        let prefix_table = Buffer::create(&context, CL_MEM_READ_ONLY, 65536, ptr::null_mut())?;
        let output = Buffer::create(&context, CL_MEM_READ_WRITE, 1 + MAX_FOUND * 8, ptr::null_mut())?;
        
        Ok(Self {
            context,
            queue,
            kernel,
            start_x, start_y,
            gn_x, gn_y,
            gn2_x, gn2_y,
            prefix_table,
            output,
            num_threads,
            device_name,
            compute_units,
            keys: vec![[0u8; 32]; num_threads],
        })
    }
}

impl GpuBackend for OpenCLBackend {
    fn name(&self) -> &str {
        &self.device_name
    }
    
    fn compute_units(&self) -> u32 {
        self.compute_units
    }
    
    fn set_prefixes(&mut self, prefixes: &[String]) -> Result<()> {
        let mut table = vec![0u16; 65536];
        
        for prefix in prefixes {
            if let Some(data) = crate::crypto::bech32::decode_prefix(prefix) {
                mark_prefix_entries(&mut table, &data);
            }
        }
        
        unsafe {
            self.queue.enqueue_write_buffer(&mut self.prefix_table, true, 0, &table, &[])?;
        }
        Ok(())
    }
    
    fn set_keys(&mut self, keys: &[[u8; 32]]) -> Result<()> {
        self.keys = keys.to_vec();
        
        // Compute starting points and upload
        let mut x_data = vec![0u64; self.num_threads * 4];
        let mut y_data = vec![0u64; self.num_threads * 4];
        
        // ... compute points
        
        unsafe {
            self.queue.enqueue_write_buffer(&mut self.start_x, true, 0, &x_data, &[])?;
            self.queue.enqueue_write_buffer(&mut self.start_y, true, 0, &y_data, &[])?;
        }
        
        Ok(())
    }
    
    fn launch(&mut self) -> Result<Vec<GpuFoundItem>> {
        // Clear output
        let zeros = vec![0u32; 1];
        unsafe {
            self.queue.enqueue_write_buffer(&mut self.output, true, 0, &zeros, &[])?;
        }
        
        // Set kernel args and execute
        ExecuteKernel::new(&self.kernel)
            .set_arg(&self.start_x)
            .set_arg(&self.start_y)
            .set_arg(&self.gn_x)
            .set_arg(&self.gn_y)
            .set_arg(&self.gn2_x)
            .set_arg(&self.gn2_y)
            .set_arg(&self.prefix_table)
            .set_arg(&self.output)
            .set_arg(&(MAX_FOUND as u32))
            .set_global_work_size(self.num_threads)
            .set_local_work_size(128)
            .enqueue_nd_range(&self.queue)?;
        
        self.queue.finish()?;
        
        // Read results
        let mut output = vec![0u32; 1 + MAX_FOUND * 8];
        unsafe {
            self.queue.enqueue_read_buffer(&self.output, true, 0, &mut output, &[])?;
        }
        
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
        // Track key advancement
    }
}

fn mark_prefix_entries(table: &mut [u16], prefix_data: &[u8]) {
    // Same as CUDA version
    let bits = prefix_data.len() * 5;
    if bits >= 16 {
        let prefix16 = (prefix_data[0] as u16) << 11 
                     | (prefix_data.get(1).copied().unwrap_or(0) as u16) << 6
                     | (prefix_data.get(2).copied().unwrap_or(0) as u16) << 1;
        table[prefix16 as usize] = 1;
    } else {
        let num_entries = 1usize << (16 - bits);
        let base = prefix_data.iter()
            .enumerate()
            .fold(0u16, |acc, (i, &v)| acc | ((v as u16) << (11 - i * 5)));
        for i in 0..num_entries {
            table[(base as usize | i)] = 1;
        }
    }
}
```

## src/gpu/opencl/kernels/vanity.cl

```c
// OpenCL kernel - same algorithm as CUDA
// See new_project_instructions for full implementation

#ifndef GRP_SIZE
#define GRP_SIZE 128
#endif

#ifndef MAX_FOUND
#define MAX_FOUND 64
#endif

typedef ulong uint64_t;
typedef uint uint32_t;
typedef ushort uint16_t;
typedef uchar uint8_t;

// ... (Include math256.cl, secp256k1.cl, hash.cl content)

__kernel void VanitySearch(
    __global const ulong *startX,
    __global const ulong *startY,
    __global const ulong *GnX,
    __global const ulong *GnY,
    __global const ulong *_2GnX,
    __global const ulong *_2GnY,
    __global const ushort *prefix_table,
    __global uint *output,
    uint maxFound
) {
    uint tid = get_global_id(0);
    
    // Load starting point
    ulong sx[4], sy[4];
    for (int i = 0; i < 4; i++) {
        sx[i] = startX[tid * 4 + i];
        sy[i] = startY[tid * 4 + i];
    }
    
    // ... (Full kernel implementation from C# docs)
}
```
