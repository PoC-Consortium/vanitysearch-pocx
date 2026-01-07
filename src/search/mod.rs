//! Search engines module

pub mod cpu;

#[cfg(feature = "cuda")]
pub mod gpu;

#[cfg(feature = "opencl")]
pub mod opencl;

pub use cpu::{CpuSearchConfig, CpuSearchEngine, Match};

#[cfg(feature = "cuda")]
pub use gpu::{GpuSearchConfig, GpuSearchEngine};

#[cfg(feature = "opencl")]
pub use opencl::{OpenClSearchConfig, OpenClSearchEngine};
