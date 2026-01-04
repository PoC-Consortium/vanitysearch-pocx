# Rust: GPU Detection und Backend Abstraction

## src/gpu/mod.rs

```rust
pub mod detect;
#[cfg(feature = "cuda")]
pub mod cuda;
pub mod opencl;

use crate::result::GpuFoundItem;
use anyhow::Result;

/// Unified GPU backend trait
pub trait GpuBackend: Send + Sync {
    fn name(&self) -> &str;
    fn compute_units(&self) -> u32;
    
    /// Set search prefixes (builds lookup table)
    fn set_prefixes(&mut self, prefixes: &[String]) -> Result<()>;
    
    /// Set starting keys for all threads
    fn set_keys(&mut self, keys: &[[u8; 32]]) -> Result<()>;
    
    /// Execute one iteration, returns found items
    fn launch(&mut self) -> Result<Vec<GpuFoundItem>>;
    
    /// Number of keys checked per launch
    fn keys_per_launch(&self) -> u64;
    
    /// Advance all keys by keys_per_launch
    fn advance_keys(&mut self);
}

pub use detect::{detect_gpu, GpuType};
```

## src/gpu/detect.rs

```rust
use anyhow::{Result, anyhow};

#[derive(Debug, Clone)]
pub enum GpuType {
    Cuda { name: String, compute_capability: (u32, u32) },
    OpenCL { name: String, vendor: String },
}

#[derive(Debug)]
pub struct GpuInfo {
    pub index: u32,
    pub gpu_type: GpuType,
    pub compute_units: u32,
    pub memory_mb: u64,
}

/// Detect available GPUs, preferring CUDA for NVIDIA
pub fn detect_gpu(device_id: u32, force_backend: Option<&str>) -> Result<Box<dyn super::GpuBackend>> {
    match force_backend {
        Some("cuda") => {
            #[cfg(feature = "cuda")]
            return super::cuda::CudaBackend::new(device_id).map(|b| Box::new(b) as _);
            #[cfg(not(feature = "cuda"))]
            return Err(anyhow!("CUDA support not compiled"));
        }
        Some("opencl") => {
            return super::opencl::OpenCLBackend::new(device_id).map(|b| Box::new(b) as _);
        }
        _ => {}
    }
    
    // Auto-detect: try CUDA first (NVIDIA only)
    #[cfg(feature = "cuda")]
    if let Ok(backend) = super::cuda::CudaBackend::new(device_id) {
        log::info!("Using CUDA backend: {}", backend.name());
        return Ok(Box::new(backend));
    }
    
    // Fallback to OpenCL
    if let Ok(backend) = super::opencl::OpenCLBackend::new(device_id) {
        log::info!("Using OpenCL backend: {}", backend.name());
        return Ok(Box::new(backend));
    }
    
    Err(anyhow!("No GPU backend available"))
}

/// List all available GPUs
pub fn list_gpus() -> Vec<GpuInfo> {
    let mut gpus = Vec::new();
    let mut index = 0;
    
    // CUDA devices
    #[cfg(feature = "cuda")]
    if let Ok(count) = cudarc::driver::result::device::get_count() {
        for i in 0..count {
            if let Ok(device) = cudarc::driver::CudaDevice::new(i) {
                gpus.push(GpuInfo {
                    index,
                    gpu_type: GpuType::Cuda {
                        name: device.name().unwrap_or_default(),
                        compute_capability: device.compute_capability(),
                    },
                    compute_units: device.num_sms() as u32,
                    memory_mb: device.total_memory().unwrap_or(0) / (1024 * 1024),
                });
                index += 1;
            }
        }
    }
    
    // OpenCL devices (skip NVIDIA if CUDA available)
    if let Ok(platforms) = opencl3::platform::get_platforms() {
        for platform in platforms {
            if let Ok(devices) = platform.get_devices(opencl3::device::CL_DEVICE_TYPE_GPU) {
                for device in devices {
                    let vendor = device.vendor().unwrap_or_default();
                    
                    // Skip NVIDIA if we have CUDA
                    #[cfg(feature = "cuda")]
                    if vendor.to_lowercase().contains("nvidia") && index > 0 {
                        continue;
                    }
                    
                    gpus.push(GpuInfo {
                        index,
                        gpu_type: GpuType::OpenCL {
                            name: device.name().unwrap_or_default(),
                            vendor,
                        },
                        compute_units: device.max_compute_units().unwrap_or(0),
                        memory_mb: device.global_mem_size().unwrap_or(0) / (1024 * 1024),
                    });
                    index += 1;
                }
            }
        }
    }
    
    gpus
}
```
