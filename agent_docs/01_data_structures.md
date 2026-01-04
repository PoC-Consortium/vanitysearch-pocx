# Rust: Datenstrukturen und Config

## src/config.rs

```rust
use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "pocx-vanity")]
#[command(about = "PoCX Bech32 Vanity Address Generator")]
pub struct Config {
    /// Prefix to search (e.g., "pocx1qtest")
    #[arg(short, long, required = true)]
    pub prefix: Vec<String>,

    /// Force specific backend: "cuda" or "opencl"
    #[arg(long)]
    pub backend: Option<String>,

    /// GPU device ID
    #[arg(short, long, default_value = "0")]
    pub gpu: u32,

    /// Stop after finding N addresses
    #[arg(short, long, default_value = "1")]
    pub max_found: u32,

    /// Output file (stdout if not specified)
    #[arg(short, long)]
    pub output: Option<String>,

    /// Seed for deterministic key generation
    #[arg(long)]
    pub seed: Option<String>,

    /// Grid size (auto if not specified)
    #[arg(long)]
    pub grid_size: Option<u32>,

    /// List available GPUs
    #[arg(long)]
    pub list_gpus: bool,
}
```

## src/result.rs

```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub address_mainnet: String,
    pub address_testnet: String,
    pub private_key_hex: String,
    pub wif_mainnet: String,
    pub wif_testnet: String,
    pub public_key_hex: String,
}

impl std::fmt::Display for SearchResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, r#"
========== FOUND ==========
Address (Mainnet): {}
Address (Testnet): {}
Private Key (HEX): 0x{}
WIF (Mainnet):     {}
WIF (Testnet):     {}
Public Key:        {}
==========================="#,
            self.address_mainnet,
            self.address_testnet,
            self.private_key_hex,
            self.wif_mainnet,
            self.wif_testnet,
            self.public_key_hex
        )
    }
}

/// GPU kernel result (matches kernel output structure)
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuFoundItem {
    pub thread_id: u32,
    pub increment: i32,
    pub endomorphism: i32,
    pub hash160: [u8; 20],
}
```

## src/crypto/mod.rs

```rust
pub mod secp256k1;
pub mod bech32;

pub use self::secp256k1::*;
pub use self::bech32::*;

/// Hash160 = RIPEMD160(SHA256(data))
pub fn hash160(data: &[u8]) -> [u8; 20] {
    use ripemd::Ripemd160;
    use sha2::{Sha256, Digest};
    
    let sha256 = Sha256::digest(data);
    let ripemd = Ripemd160::digest(&sha256);
    
    let mut result = [0u8; 20];
    result.copy_from_slice(&ripemd);
    result
}
```
