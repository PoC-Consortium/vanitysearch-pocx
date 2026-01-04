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
=========================="#,
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
    pub _padding: u32,
    pub hash160: [u8; 20],
}
