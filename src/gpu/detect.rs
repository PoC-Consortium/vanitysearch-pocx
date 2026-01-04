use anyhow::{Result, anyhow};
use super::GpuBackend;

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum GpuType {
    Cuda { name: String, compute_capability: (u32, u32) },
    OpenCL { name: String, vendor: String },
}

#[derive(Debug)]
#[allow(dead_code)]
pub struct GpuInfo {
    pub index: u32,
    pub gpu_type: GpuType,
    pub compute_units: u32,
    pub memory_mb: u64,
}

/// Detect available GPUs, preferring CUDA for NVIDIA
#[allow(unused_variables)]
pub fn detect_gpu(device_id: u32, force_backend: Option<&str>) -> Result<Box<dyn GpuBackend>> {
    match force_backend {
        Some("cuda") => {
            #[cfg(feature = "cuda")]
            return super::cuda::CudaBackend::new(device_id).map(|b| Box::new(b) as Box<dyn GpuBackend>);
            #[cfg(not(feature = "cuda"))]
            return Err(anyhow!("CUDA support not compiled"));
        }
        Some("opencl") => {
            #[cfg(feature = "opencl")]
            return super::opencl::OpenCLBackend::new(device_id).map(|b| Box::new(b) as Box<dyn GpuBackend>);
            #[cfg(not(feature = "opencl"))]
            return Err(anyhow!("OpenCL support not compiled"));
        }
        _ => {}
    }
    
    // Auto-detect: try CUDA first (NVIDIA only)
    #[cfg(feature = "cuda")]
    if let Ok(backend) = super::cuda::CudaBackend::new(device_id) {
        log::info!("Using CUDA backend: {}", GpuBackend::name(&backend));
        return Ok(Box::new(backend));
    }
    
    // Fallback to OpenCL
    #[cfg(feature = "opencl")]
    if let Ok(backend) = super::opencl::OpenCLBackend::new(device_id) {
        log::info!("Using OpenCL backend: {}", GpuBackend::name(&backend));
        return Ok(Box::new(backend));
    }
    
    Err(anyhow!("No GPU backend available. Compile with 'cuda' or 'opencl' feature."))
}

/// List all available GPUs
#[allow(unused_mut, unused_variables)]
pub fn list_gpus() -> Vec<GpuInfo> {
    let mut gpus = Vec::new();
    let mut index = 0u32;
    
    // CUDA devices
    #[cfg(feature = "cuda")]
    {
        use cudarc::driver::result::device::get_count;
        use cudarc::driver::CudaDevice;
        
        if let Ok(count) = get_count() {
            for i in 0..count {
                if let Ok(device) = CudaDevice::new(i) {
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
    }
    
    // OpenCL devices (skip NVIDIA if CUDA available)
    #[cfg(feature = "opencl")]
    {
        if let Ok(platforms) = opencl3::platform::get_platforms() {
            for platform in platforms {
                if let Ok(devices) = platform.get_devices(opencl3::device::CL_DEVICE_TYPE_GPU) {
                    for device_id in devices {
                        let device = opencl3::device::Device::from(device_id);
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
    }
    
    gpus
}
