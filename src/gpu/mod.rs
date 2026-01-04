pub mod detect;
#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "opencl")]
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

pub use detect::{detect_gpu, list_gpus, GpuType};
