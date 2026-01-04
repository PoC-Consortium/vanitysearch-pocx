use secp256k1::{Secp256k1, SecretKey, PublicKey};
use sha2::{Sha256, Digest};

/// Secp256k1 constants for GPU
#[allow(dead_code)]
pub mod constants {
    /// Prime field p = 2^256 - 2^32 - 977
    pub const P: [u64; 4] = [
        0xFFFFFFFFFFFFFC2F, 0xFFFFFFFFFFFFFFFE,
        0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF,
    ];
    
    /// Curve order n
    pub const N: [u64; 4] = [
        0xBFD25E8CD0364141, 0xBAAEDCE6AF48A03B,
        0xFFFFFFFFFFFFFFFE, 0xFFFFFFFFFFFFFFFF,
    ];
    
    /// Generator X
    pub const GX: [u64; 4] = [
        0x59F2815B16F81798, 0x029BFCDB2DCE28D9,
        0x55A06295CE870B07, 0x79BE667EF9DCBBAC,
    ];
    
    /// Generator Y
    pub const GY: [u64; 4] = [
        0x9C47D08FFB10D4B8, 0xFD17B448A6855419,
        0x5DA4FBFC0E1108A8, 0x483ADA7726A3C465,
    ];
    
    /// Endomorphism beta
    pub const BETA: [u64; 4] = [
        0xC1396C28719501EE, 0x9CF0497512F58995,
        0x6E64479EAC3434E9, 0x7AE96A2B657C0710,
    ];
    
    /// Endomorphism lambda
    pub const LAMBDA: [u64; 4] = [
        0xDF02967C1B23BD72, 0x122E22EA20816678,
        0xA5261C028812645A, 0x5363AD4CC05C30E0,
    ];
}

pub struct Crypto {
    secp: Secp256k1<secp256k1::All>,
}

impl Default for Crypto {
    fn default() -> Self {
        Self::new()
    }
}

impl Crypto {
    pub fn new() -> Self {
        Self { secp: Secp256k1::new() }
    }
    
    /// Get compressed public key from private key bytes
    pub fn public_key(&self, secret: &[u8; 32]) -> Result<[u8; 33], secp256k1::Error> {
        let sk = SecretKey::from_slice(secret)?;
        let pk = PublicKey::from_secret_key(&self.secp, &sk);
        Ok(pk.serialize())
    }
    
    /// Recover private key from GPU result
    pub fn recover_private_key(
        &self,
        base_key: &[u8; 32],
        increment: i32,
        endomorphism: i32,
    ) -> [u8; 32] {
        use num_bigint::BigUint;
        
        let n = BigUint::from_bytes_be(&hex::decode(
            "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141"
        ).unwrap());
        
        let lambda = BigUint::from_bytes_be(&hex::decode(
            "5363AD4CC05C30E0A5261C028812645A122E22EA20816678DF02967C1B23BD72"
        ).unwrap());
        
        let lambda2 = BigUint::from_bytes_be(&hex::decode(
            "AC9C52B33FA3CF1F5AD9E3FD77ED9BA4A880B9FC8EC739C2E0CFC810B51283CE"
        ).unwrap());
        
        let mut k = BigUint::from_bytes_be(base_key);
        
        // Apply increment
        if increment < 0 {
            k = (k + BigUint::from(-increment as u32)) % &n;
            k = &n - k;
        } else {
            k = (k + BigUint::from(increment as u32)) % &n;
        }
        
        // Apply endomorphism
        k = match endomorphism {
            1 => (k * &lambda) % &n,
            2 => (k * &lambda2) % &n,
            _ => k,
        };
        
        let mut result = [0u8; 32];
        let bytes = k.to_bytes_be();
        result[32 - bytes.len()..].copy_from_slice(&bytes);
        result
    }
    
    /// Generate WIF (Wallet Import Format)
    pub fn to_wif(&self, secret: &[u8; 32], mainnet: bool) -> String {
        let version = if mainnet { 0x80 } else { 0xEF };
        
        let mut data = vec![version];
        data.extend_from_slice(secret);
        data.push(0x01); // compressed
        
        let checksum = &Sha256::digest(&Sha256::digest(&data))[..4];
        data.extend_from_slice(checksum);
        
        bs58::encode(data).into_string()
    }
}
