use bech32::{self, ToBase32, FromBase32, Variant};

pub const HRP_MAINNET: &str = "pocx";
pub const HRP_TESTNET: &str = "tpocx";

/// Encode Hash160 to Bech32 address
pub fn encode(hash160: &[u8; 20], mainnet: bool) -> String {
    let hrp = if mainnet { HRP_MAINNET } else { HRP_TESTNET };
    
    let mut data = vec![bech32::u5::try_from_u8(0).unwrap()]; // witness version 0
    data.extend(hash160.to_base32());
    
    bech32::encode(hrp, data, Variant::Bech32).unwrap()
}

/// Decode Bech32 address to Hash160
pub fn decode(address: &str) -> Option<[u8; 20]> {
    let (hrp, data, _) = bech32::decode(address).ok()?;
    
    if hrp != HRP_MAINNET && hrp != HRP_TESTNET {
        return None;
    }
    
    if data.is_empty() || data[0].to_u8() != 0 {
        return None;
    }
    
    let hash = Vec::<u8>::from_base32(&data[1..]).ok()?;
    if hash.len() != 20 {
        return None;
    }
    
    let mut result = [0u8; 20];
    result.copy_from_slice(&hash);
    Some(result)
}

/// Decode prefix to 5-bit values for matching
pub fn decode_prefix(prefix: &str) -> Option<Vec<u8>> {
    let prefix = prefix.to_lowercase();
    
    let data_start = if prefix.starts_with("pocx1q") {
        6
    } else if prefix.starts_with("tpocx1q") {
        7
    } else {
        return None;
    };
    
    let charset = "qpzry9x8gf2tvdw0s3jn54khce6mua7l";
    
    prefix[data_start..]
        .chars()
        .map(|c| charset.find(c).map(|i| i as u8))
        .collect()
}

/// Calculate difficulty for prefix
pub fn difficulty(prefix: &str) -> f64 {
    let bits = match decode_prefix(prefix) {
        Some(data) => data.len() * 5,
        None => 0,
    };
    2f64.powi(bits as i32)
}
