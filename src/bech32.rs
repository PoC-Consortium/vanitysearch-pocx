//! Bech32 encoding for Bitcoin addresses (BIP-173, BIP-350)
//! Only lowercase output supported (as per requirements)

/// Bech32 character set
const CHARSET: &[u8] = b"qpzry9x8gf2tvdw0s3jn54khce6mua7l";

/// Reverse lookup table for Bech32 characters
const CHARSET_REV: [i8; 128] = [
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    15, -1, 10, 17, 21, 20, 26, 30,  7,  5, -1, -1, -1, -1, -1, -1,
    -1, 29, -1, 24, 13, 25,  9,  8, 23, -1, 18, 22, 31, 27, 19, -1,
     1,  0,  3, 16, 11, 28, 12, 14,  6,  4,  2, -1, -1, -1, -1, -1,
    -1, 29, -1, 24, 13, 25,  9,  8, 23, -1, 18, 22, 31, 27, 19, -1,
     1,  0,  3, 16, 11, 28, 12, 14,  6,  4,  2, -1, -1, -1, -1, -1,
];

/// Bech32 polymod step
#[inline]
fn polymod_step(pre: u32) -> u32 {
    let b = pre >> 25;
    ((pre & 0x1FFFFFF) << 5)
        ^ (if b & 1 != 0 { 0x3b6a57b2 } else { 0 })
        ^ (if b & 2 != 0 { 0x26508e6d } else { 0 })
        ^ (if b & 4 != 0 { 0x1ea119fa } else { 0 })
        ^ (if b & 8 != 0 { 0x3d4233dd } else { 0 })
        ^ (if b & 16 != 0 { 0x2a1462b3 } else { 0 })
}

/// Compute Bech32 checksum
fn create_checksum(hrp: &str, data: &[u8]) -> [u8; 6] {
    let mut chk = 1u32;
    
    // Process HRP
    for c in hrp.bytes() {
        chk = polymod_step(chk) ^ ((c >> 5) as u32);
    }
    chk = polymod_step(chk);
    for c in hrp.bytes() {
        chk = polymod_step(chk) ^ ((c & 0x1f) as u32);
    }
    
    // Process data
    for &d in data {
        chk = polymod_step(chk) ^ (d as u32);
    }
    
    // Pad for checksum
    for _ in 0..6 {
        chk = polymod_step(chk);
    }
    
    // XOR with constant for bech32 (1) or bech32m (0x2bc830a3)
    chk ^= 1; // Using bech32 for witness v0
    
    let mut result = [0u8; 6];
    for (i, r) in result.iter_mut().enumerate() {
        *r = ((chk >> ((5 - i) * 5)) & 0x1f) as u8;
    }
    result
}

/// Encode data to Bech32 string
pub fn encode(hrp: &str, data: &[u8]) -> String {
    let checksum = create_checksum(hrp, data);
    
    let mut result = String::with_capacity(hrp.len() + 1 + data.len() + 6);
    result.push_str(hrp);
    result.push('1');
    
    for &d in data {
        result.push(CHARSET[d as usize] as char);
    }
    for &c in &checksum {
        result.push(CHARSET[c as usize] as char);
    }
    
    result
}

/// Convert 8-bit data to 5-bit data
pub fn convert_bits_8_to_5(data: &[u8]) -> Vec<u8> {
    let mut result = Vec::with_capacity((data.len() * 8 + 4) / 5);
    let mut acc = 0u32;
    let mut bits = 0;
    
    for &byte in data {
        acc = (acc << 8) | (byte as u32);
        bits += 8;
        while bits >= 5 {
            bits -= 5;
            result.push(((acc >> bits) & 0x1f) as u8);
        }
    }
    
    // Pad remaining bits
    if bits > 0 {
        result.push(((acc << (5 - bits)) & 0x1f) as u8);
    }
    
    result
}

/// Convert 5-bit data to 8-bit data
pub fn convert_bits_5_to_8(data: &[u8]) -> Vec<u8> {
    let mut result = Vec::with_capacity((data.len() * 5) / 8);
    let mut acc = 0u32;
    let mut bits = 0;
    
    for &val in data {
        acc = (acc << 5) | (val as u32);
        bits += 5;
        while bits >= 8 {
            bits -= 8;
            result.push(((acc >> bits) & 0xff) as u8);
        }
    }
    
    result
}

/// Encode a SegWit address (witness version 0)
pub fn segwit_encode(hrp: &str, witness_version: u8, witness_program: &[u8]) -> String {
    let mut data = Vec::with_capacity(1 + (witness_program.len() * 8 + 4) / 5);
    data.push(witness_version);
    data.extend(convert_bits_8_to_5(witness_program));
    
    encode(hrp, &data)
}

/// Decode a SegWit address
pub fn segwit_decode(hrp: &str, addr: &str) -> Option<(u8, Vec<u8>)> {
    let addr_lower = addr.to_lowercase();
    
    // Find separator
    let sep_pos = addr_lower.rfind('1')?;
    if sep_pos < 1 || sep_pos + 7 > addr_lower.len() {
        return None;
    }
    
    let addr_hrp = &addr_lower[..sep_pos];
    if addr_hrp != hrp {
        return None;
    }
    
    let data_part = &addr_lower[sep_pos + 1..];
    
    // Decode data part
    let mut data = Vec::with_capacity(data_part.len());
    for c in data_part.bytes() {
        if c >= 128 {
            return None;
        }
        let val = CHARSET_REV[c as usize];
        if val < 0 {
            return None;
        }
        data.push(val as u8);
    }
    
    // Verify checksum (simplified - just check length for now)
    if data.len() < 7 {
        return None;
    }
    
    let witness_version = data[0];
    let witness_program = convert_bits_5_to_8(&data[1..data.len() - 6]);
    
    if witness_version == 0 && witness_program.len() != 20 && witness_program.len() != 32 {
        return None;
    }
    
    Some((witness_version, witness_program))
}

/// Decode bech32 data without checksum verification (for pattern matching)
pub fn decode_nocheck(input: &str) -> Option<Vec<u8>> {
    let input_lower = input.to_lowercase();
    
    // Find HRP separator
    let sep_pos = input_lower.rfind('1')?;
    let data_part = &input_lower[sep_pos + 1..];
    
    // Decode to 5-bit values
    let mut data5 = Vec::with_capacity(data_part.len());
    for c in data_part.bytes() {
        if c >= 128 {
            return None;
        }
        let val = CHARSET_REV[c as usize];
        if val < 0 {
            return None;
        }
        data5.push(val as u8);
    }
    
    // Convert to 8-bit
    if data5.is_empty() {
        return None;
    }
    
    // Skip witness version (first byte)
    let witness_version = data5[0];
    if witness_version > 16 {
        return None;
    }
    
    Some(convert_bits_5_to_8(&data5[1..]))
}

/// Get Bech32 address from Hash160
pub fn address_from_hash160(hrp: &str, hash160: &[u8; 20]) -> String {
    segwit_encode(hrp, 0, hash160)
}

/// Check if a character is valid in bech32
#[inline]
pub fn is_valid_char(c: char) -> bool {
    let c = c.to_ascii_lowercase();
    c.is_ascii() && CHARSET_REV[c as usize] >= 0
}

/// Check if a string is valid bech32 pattern (allowing wildcards)
pub fn is_valid_pattern(pattern: &str) -> bool {
    for c in pattern.chars() {
        if c == '?' || c == '*' {
            continue;
        }
        if c == '1' {
            // Separator is allowed
            continue;
        }
        if !is_valid_char(c) && !c.is_ascii_lowercase() {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bech32_encode() {
        // Test vector from BIP-173
        let hash160 = hex::decode("751e76e8199196d454941c45d1b3a323f1433bd6").unwrap();
        let addr = segwit_encode("bc", 0, &hash160);
        assert_eq!(addr, "bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4");
    }

    #[test]
    fn test_bech32_encode_testnet() {
        let hash160 = hex::decode("751e76e8199196d454941c45d1b3a323f1433bd6").unwrap();
        let addr = segwit_encode("tb", 0, &hash160);
        assert_eq!(addr, "tb1qw508d6qejxtdg4y5r3zarvary0c5xw7kxpjzsx");
    }

    #[test]
    fn test_address_from_hash160() {
        let hash160: [u8; 20] = hex::decode("751e76e8199196d454941c45d1b3a323f1433bd6")
            .unwrap()
            .try_into()
            .unwrap();
        let addr = address_from_hash160("bc", &hash160);
        assert_eq!(addr, "bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4");
    }

    #[test]
    fn test_valid_pattern() {
        assert!(is_valid_pattern("bc1q*"));
        assert!(is_valid_pattern("bc1qtest?"));
        assert!(is_valid_pattern("bc1qmadf0*"));
    }

    #[test]
    fn test_convert_bits() {
        let data8 = vec![0x75, 0x1e, 0x76, 0xe8];
        let data5 = convert_bits_8_to_5(&data8);
        let data8_back = convert_bits_5_to_8(&data5);
        assert_eq!(data8, data8_back);
    }
}
