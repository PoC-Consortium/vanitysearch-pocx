//! Hash functions

pub mod sha256;
pub mod ripemd160;

pub use sha256::{sha256, sha256d, sha256_33, sha256_32, Sha256};
pub use ripemd160::{ripemd160, ripemd160_32, Ripemd160};

/// Compute Hash160 (RIPEMD160(SHA256(data)))
/// Used for Bitcoin addresses
pub fn hash160(data: &[u8]) -> [u8; 20] {
    ripemd160(&sha256(data))
}

/// Optimized Hash160 for exactly 33-byte compressed public keys
/// ~2x faster than generic hash160() for this specific size
#[inline]
pub fn hash160_33(pubkey: &[u8; 33]) -> [u8; 20] {
    ripemd160_32(&sha256_33(pubkey))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash160() {
        // Test with a known public key
        let pubkey = hex::decode("0279BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798").unwrap();
        let hash = hash160(&pubkey);
        let expected = hex::decode("751e76e8199196d454941c45d1b3a323f1433bd6").unwrap();
        assert_eq!(&hash[..], &expected[..]);
    }
}
