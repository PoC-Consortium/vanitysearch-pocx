pub mod secp256k1_utils;
pub mod bech32_utils;

pub use self::secp256k1_utils::*;
pub use self::bech32_utils::{encode, decode, decode_prefix, difficulty, HRP_MAINNET, HRP_TESTNET};

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
