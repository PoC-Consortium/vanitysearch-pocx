//! Search engines module

pub mod cpu;

#[cfg(feature = "cuda")]
pub mod gpu;

pub use cpu::{CpuSearchConfig, CpuSearchEngine, Match};

#[cfg(feature = "cuda")]
pub use gpu::{GpuSearchConfig, GpuSearchEngine};
