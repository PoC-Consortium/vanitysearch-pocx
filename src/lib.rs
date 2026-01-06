//! VanitySearch-POCX: Bitcoin bech32 vanity address generator
//!
//! A Rust port of VanitySearch focused on bech32 addresses with CUDA support.

pub mod bech32;
pub mod hash;
pub mod output;
pub mod pattern;
pub mod search;
pub mod secp256k1;
pub mod wif;

#[cfg(test)]
mod tests;

pub use output::{FormattedMatch, NetworkInfo, Stats};
pub use pattern::Pattern;
pub use search::{CpuSearchConfig, CpuSearchEngine, Match};
#[cfg(feature = "cuda")]
pub use search::{GpuSearchConfig, GpuSearchEngine};
