//! VanitySearch-POCX: Bitcoin bech32 vanity address generator
//!
//! A Rust port of VanitySearch focused on bech32 addresses with CUDA support.

pub mod secp256k1;
pub mod hash;
pub mod bech32;
pub mod config;
pub mod wif;
pub mod pattern;
pub mod search;
pub mod output;

#[cfg(test)]
mod tests;

pub use config::{Network, NetworksConfig};
pub use pattern::Pattern;
pub use search::{CpuSearchConfig, CpuSearchEngine, Match};
#[cfg(feature = "cuda")]
pub use search::{GpuSearchConfig, GpuSearchEngine};
pub use output::{FormattedMatch, Stats};
