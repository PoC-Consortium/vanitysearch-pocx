//! WIF (Wallet Import Format) encoding and descriptor checksum

use crate::hash::sha256d;
use crate::secp256k1::Scalar;

const BASE58_ALPHABET: &[u8] = b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

/// Encode bytes to Base58
fn encode_base58(data: &[u8]) -> String {
    // Count leading zeros
    let mut zeros = 0;
    for &byte in data {
        if byte == 0 {
            zeros += 1;
        } else {
            break;
        }
    }

    // Convert to base58
    let mut result = Vec::new();
    let mut bytes = data.to_vec();

    while !bytes.iter().all(|&b| b == 0) {
        let mut remainder = 0u32;
        for byte in bytes.iter_mut() {
            let value = (remainder << 8) + (*byte as u32);
            *byte = (value / 58) as u8;
            remainder = value % 58;
        }
        result.push(BASE58_ALPHABET[remainder as usize]);
        
        // Remove leading zeros from bytes
        while !bytes.is_empty() && bytes[0] == 0 {
            bytes.remove(0);
        }
    }

    // Add leading '1's for leading zeros
    result.extend(std::iter::repeat_n(b'1', zeros));

    result.reverse();
    String::from_utf8(result).unwrap()
}

/// Encode bytes to Base58Check (with checksum)
fn encode_base58check(data: &[u8]) -> String {
    let checksum = sha256d(data);
    let mut full = data.to_vec();
    full.extend_from_slice(&checksum[..4]);
    encode_base58(&full)
}

/// Encode private key to WIF format
pub fn encode_wif(private_key: &Scalar, compressed: bool, mainnet: bool) -> String {
    let mut data = Vec::with_capacity(38);
    
    // Version byte
    if mainnet {
        data.push(0x80);
    } else {
        data.push(0xEF);
    }
    
    // Private key bytes (32 bytes, big-endian)
    data.extend_from_slice(&private_key.to_bytes());
    
    // Compression flag
    if compressed {
        data.push(0x01);
    }
    
    encode_base58check(&data)
}

/// Decode WIF to private key bytes (returns key bytes, compressed flag, mainnet flag)
pub fn decode_wif(wif: &str) -> Option<([u8; 32], bool, bool)> {
    // Decode base58
    let mut result = vec![0u8; 64];
    let mut result_len = 0;
    
    for c in wif.chars() {
        let mut carry = match BASE58_ALPHABET.iter().position(|&x| x == c as u8) {
            Some(pos) => pos as u32,
            None => return None,
        };
        
        for byte in result[..result_len].iter_mut().rev() {
            let value = (*byte as u32) * 58 + carry;
            *byte = (value & 0xFF) as u8;
            carry = value >> 8;
        }
        
        while carry > 0 {
            result_len += 1;
            if result_len > result.len() {
                result.resize(result_len, 0);
            }
            result[result_len - 1] = 0;
            for byte in result[..result_len].iter_mut().rev() {
                let value = (*byte as u32) * 58 + carry;
                *byte = (value & 0xFF) as u8;
                carry = value >> 8;
            }
        }
    }
    
    // Add leading zeros
    for c in wif.chars() {
        if c == '1' {
            result_len += 1;
            result.resize(result_len, 0);
            result.rotate_right(1);
            result[0] = 0;
        } else {
            break;
        }
    }
    
    result.truncate(result_len);
    result.reverse();
    
    // Verify checksum
    if result.len() < 5 {
        return None;
    }
    
    let checksum_start = result.len() - 4;
    let checksum = sha256d(&result[..checksum_start]);
    if checksum[..4] != result[checksum_start..] {
        return None;
    }
    
    let data = &result[..checksum_start];
    
    // Parse version and key
    let mainnet = match data[0] {
        0x80 => true,
        0xEF => false,
        _ => return None,
    };
    
    let compressed = match data.len() {
        33 => false, // 1 version + 32 key
        34 => {
            if data[33] != 0x01 {
                return None;
            }
            true
        }
        _ => return None,
    };
    
    let mut key = [0u8; 32];
    key.copy_from_slice(&data[1..33]);
    
    Some((key, compressed, mainnet))
}

/// Descriptor checksum character set (same as bech32)
const DESCRIPTOR_CHARSET: &str = "qpzry9x8gf2tvdw0s3jn54khce6mua7l";

/// Compute descriptor checksum
pub fn descriptor_checksum(descriptor: &str) -> String {
    const INPUT_CHARSET: &str = "0123456789()[],'/*abcdefgh@:$%{}IJKLMNOPQRSTUVWXYZ&+-.;<=>?!^_|~ijklmnopqrstuvwxyzABCDEFGH`#\"\\ ";
    
    let mut c = 1u64;
    let mut cls = 0u64;
    let mut clscount = 0;
    
    for ch in descriptor.chars() {
        let pos = match INPUT_CHARSET.find(ch) {
            Some(p) => p as u64,
            None => continue, // Skip unknown characters
        };
        
        c = polymod(c, pos & 31);
        cls = cls * 3 + (pos >> 5);
        clscount += 1;
        
        if clscount == 3 {
            c = polymod(c, cls);
            cls = 0;
            clscount = 0;
        }
    }
    
    if clscount > 0 {
        c = polymod(c, cls);
    }
    
    // Finalize
    for _ in 0..8 {
        c = polymod(c, 0);
    }
    c ^= 1;
    
    // Convert to characters
    let mut result = String::with_capacity(8);
    for i in 0..8 {
        let idx = ((c >> ((7 - i) * 5)) & 31) as usize;
        result.push(DESCRIPTOR_CHARSET.chars().nth(idx).unwrap());
    }
    
    result
}

/// Polymod for descriptor checksum
fn polymod(mut c: u64, val: u64) -> u64 {
    let c0 = c >> 35;
    c = ((c & 0x7ffffffff) << 5) ^ val;
    
    if c0 & 1 != 0 { c ^= 0xf5dee51989; }
    if c0 & 2 != 0 { c ^= 0xa9fdca3312; }
    if c0 & 4 != 0 { c ^= 0x1bab10e32d; }
    if c0 & 8 != 0 { c ^= 0x3706b1677a; }
    if c0 & 16 != 0 { c ^= 0x644d626ffd; }
    
    c
}

/// Create a descriptor for a WIF key (p2wpkh)
pub fn create_descriptor(wif: &str) -> String {
    let descriptor = format!("wpkh({})", wif);
    let checksum = descriptor_checksum(&descriptor);
    format!("{}#{}", descriptor, checksum)
}

/// Create descriptors for both mainnet and testnet WIFs
pub fn create_descriptors(private_key: &Scalar, compressed: bool) -> (String, String) {
    let mainnet_wif = encode_wif(private_key, compressed, true);
    let testnet_wif = encode_wif(private_key, compressed, false);
    
    (create_descriptor(&mainnet_wif), create_descriptor(&testnet_wif))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_wif_mainnet_compressed() {
        // Test vector
        let key_bytes = hex::decode("0000000000000000000000000000000000000000000000000000000000000001").unwrap();
        let mut key = [0u64; 4];
        for i in 0..4 {
            let offset = (3 - i) * 8;
            key[i] = u64::from_be_bytes([
                key_bytes[offset], key_bytes[offset + 1], key_bytes[offset + 2], key_bytes[offset + 3],
                key_bytes[offset + 4], key_bytes[offset + 5], key_bytes[offset + 6], key_bytes[offset + 7],
            ]);
        }
        let scalar = Scalar::new(key);
        let wif = encode_wif(&scalar, true, true);
        assert_eq!(wif, "KwDiBf89QgGbjEhKnhXJuH7LrciVrZi3qYjgd9M7rFU73sVHnoWn");
    }

    #[test]
    fn test_descriptor_checksum() {
        // Test with known descriptor
        let desc = "wpkh(KwDiBf89QgGbjEhKnhXJuH7LrciVrZi3qYjgd9M7rFU73sVHnoWn)";
        let checksum = descriptor_checksum(desc);
        assert_eq!(checksum.len(), 8);
    }

    #[test]
    fn test_create_descriptor() {
        let wif = "KwDiBf89QgGbjEhKnhXJuH7LrciVrZi3qYjgd9M7rFU73sVHnoWn";
        let descriptor = create_descriptor(wif);
        assert!(descriptor.contains('#'));
        assert!(descriptor.starts_with("wpkh("));
    }
}
