use crate::gpu::GpuBackend;
use crate::result::GpuFoundItem;
use crate::crypto::bech32_utils;
use anyhow::{Result, anyhow};
use opencl3::command_queue::{CommandQueue, CL_BLOCKING};
use opencl3::context::Context;
use opencl3::device::{Device, CL_DEVICE_TYPE_GPU};
use opencl3::kernel::{Kernel, ExecuteKernel};
use opencl3::memory::{Buffer, CL_MEM_READ_WRITE, CL_MEM_READ_ONLY};
use opencl3::platform::get_platforms;
use opencl3::program::Program;
use opencl3::types::{cl_ulong, cl_uint, cl_ushort};
use std::ptr;
use std::sync::Mutex;

const GRP_SIZE: usize = 128;
const MAX_FOUND: usize = 64;

pub struct OpenCLBackend {
    _context: Context,
    queue: Mutex<CommandQueue>,
    kernel: Mutex<Kernel>,
    
    // Buffers
    start_x: Mutex<Buffer<cl_ulong>>,
    start_y: Mutex<Buffer<cl_ulong>>,
    gn_x: Buffer<cl_ulong>,
    gn_y: Buffer<cl_ulong>,
    gn2_x: Buffer<cl_ulong>,
    gn2_y: Buffer<cl_ulong>,
    prefix_table: Mutex<Buffer<cl_ushort>>,
    output: Mutex<Buffer<cl_uint>>,
    
    num_threads: usize,
    device_name: String,
    compute_units: u32,
    keys: Mutex<Vec<[u8; 32]>>,
}

impl OpenCLBackend {
    pub fn new(device_id: u32) -> Result<Self> {
        let platforms = get_platforms()?;
        
        // Find GPU device
        let mut gpu_device: Option<Device> = None;
        let mut current_id = 0u32;
        
        'outer: for platform in &platforms {
            let devices = platform.get_devices(CL_DEVICE_TYPE_GPU)?;
            for device in devices {
                if current_id == device_id {
                    gpu_device = Some(Device::from(device));
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
        let queue = CommandQueue::create_default_with_properties(&context, 0, 0)?;
        
        // Compile kernel
        let source = include_str!("kernels/vanity.cl");
        let program = Program::create_and_build_from_source(
            &context,
            source,
            &format!("-D GRP_SIZE={} -D MAX_FOUND={}", GRP_SIZE, MAX_FOUND),
        ).map_err(|e| anyhow!("Kernel compilation failed: {:?}", e))?;
        let kernel = Kernel::create(&program, "VanitySearch")?;
        
        let num_threads = (compute_units as usize) * 8;
        
        // Allocate buffers
        let start_x = unsafe { Buffer::create(&context, CL_MEM_READ_WRITE, num_threads * 4, ptr::null_mut())? };
        let start_y = unsafe { Buffer::create(&context, CL_MEM_READ_WRITE, num_threads * 4, ptr::null_mut())? };
        let gn_x = unsafe { Buffer::create(&context, CL_MEM_READ_ONLY, GRP_SIZE / 2 * 4, ptr::null_mut())? };
        let gn_y = unsafe { Buffer::create(&context, CL_MEM_READ_ONLY, GRP_SIZE / 2 * 4, ptr::null_mut())? };
        let gn2_x = unsafe { Buffer::create(&context, CL_MEM_READ_ONLY, 4, ptr::null_mut())? };
        let gn2_y = unsafe { Buffer::create(&context, CL_MEM_READ_ONLY, 4, ptr::null_mut())? };
        let prefix_table = unsafe { Buffer::create(&context, CL_MEM_READ_ONLY, 65536, ptr::null_mut())? };
        let output = unsafe { Buffer::create(&context, CL_MEM_READ_WRITE, 1 + MAX_FOUND * 8, ptr::null_mut())? };
        
        Ok(Self {
            _context: context,
            queue: Mutex::new(queue),
            kernel: Mutex::new(kernel),
            start_x: Mutex::new(start_x),
            start_y: Mutex::new(start_y),
            gn_x, gn_y,
            gn2_x, gn2_y,
            prefix_table: Mutex::new(prefix_table),
            output: Mutex::new(output),
            num_threads,
            device_name,
            compute_units,
            keys: Mutex::new(vec![[0u8; 32]; num_threads]),
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
            if let Some(data) = bech32_utils::decode_prefix(prefix) {
                mark_prefix_entries(&mut table, &data);
            }
        }
        
        let queue = self.queue.lock().unwrap();
        let mut prefix_table = self.prefix_table.lock().unwrap();
        unsafe {
            queue.enqueue_write_buffer(&mut prefix_table, CL_BLOCKING, 0, &table, &[])?;
        }
        Ok(())
    }
    
    fn set_keys(&mut self, keys: &[[u8; 32]]) -> Result<()> {
        *self.keys.lock().unwrap() = keys.to_vec();
        
        // Compute starting points and upload
        let x_data = vec![0u64; self.num_threads * 4];
        let y_data = vec![0u64; self.num_threads * 4];
        
        // ... compute points
        
        let queue = self.queue.lock().unwrap();
        let mut start_x = self.start_x.lock().unwrap();
        let mut start_y = self.start_y.lock().unwrap();
        unsafe {
            queue.enqueue_write_buffer(&mut start_x, CL_BLOCKING, 0, &x_data, &[])?;
            queue.enqueue_write_buffer(&mut start_y, CL_BLOCKING, 0, &y_data, &[])?;
        }
        
        Ok(())
    }
    
    fn launch(&mut self) -> Result<Vec<GpuFoundItem>> {
        // Clear output
        let zeros = vec![0u32; 1];
        let queue = self.queue.lock().unwrap();
        let mut output_buf = self.output.lock().unwrap();
        unsafe {
            queue.enqueue_write_buffer(&mut output_buf, CL_BLOCKING, 0, &zeros, &[])?;
        }
        
        // Set kernel args and execute
        let max_found_arg = MAX_FOUND as u32;
        let kernel = self.kernel.lock().unwrap();
        let start_x = self.start_x.lock().unwrap();
        let start_y = self.start_y.lock().unwrap();
        let prefix_table = self.prefix_table.lock().unwrap();
        unsafe {
            ExecuteKernel::new(&kernel)
                .set_arg(&*start_x)
                .set_arg(&*start_y)
                .set_arg(&self.gn_x)
                .set_arg(&self.gn_y)
                .set_arg(&self.gn2_x)
                .set_arg(&self.gn2_y)
                .set_arg(&*prefix_table)
                .set_arg(&*output_buf)
                .set_arg(&max_found_arg)
                .set_global_work_size(self.num_threads)
                .set_local_work_size(128)
                .enqueue_nd_range(&queue)?;
        }
        
        queue.finish()?;
        
        // Read results
        let mut output = vec![0u32; 1 + MAX_FOUND * 8];
        unsafe {
            queue.enqueue_read_buffer(&output_buf, CL_BLOCKING, 0, &mut output, &[])?;
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
            table[base as usize | i] = 1;
        }
    }
}
