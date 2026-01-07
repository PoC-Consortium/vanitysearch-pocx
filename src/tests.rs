//! Unit tests for the bech32 vanity search pipeline
//! Tests each step: SHA256 -> RIPEMD160 -> Hash160 -> Bech32 5-bit -> Pattern matching

#[cfg(test)]
mod tests {
    use crate::bech32;
    use crate::hash::{ripemd160, sha256};
    use crate::pattern::{FastPrefixMatcher, Pattern};
    use crate::secp256k1::{Scalar, G};

    /// Known test vector: private key -> public key -> address
    /// Private key: 0x0000000000000000000000000000000000000000000000000000000000000001
    /// Expected public key (compressed): 0279BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
    /// Expected address: bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4
    const TEST_PRIVKEY_1: [u64; 4] = [1, 0, 0, 0];
    const TEST_PUBKEY_COMPRESSED_1: [u8; 33] = [
        0x02, 0x79, 0xBE, 0x66, 0x7E, 0xF9, 0xDC, 0xBB, 0xAC, 0x55, 0xA0, 0x62, 0x95, 0xCE, 0x87,
        0x0B, 0x07, 0x02, 0x9B, 0xFC, 0xDB, 0x2D, 0xCE, 0x28, 0xD9, 0x59, 0xF2, 0x81, 0x5B, 0x16,
        0xF8, 0x17, 0x98,
    ];
    const TEST_ADDRESS_1: &str = "bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4";

    // Test vector 2: Random key for validation
    // Private key: 12345 (0x3039)
    // This is a known-good test vector generated with Python
    const TEST_PRIVKEY_2: [u8; 32] = [
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x30, 0x39,
    ];
    const TEST_ADDRESS_2: &str = "bc1qz5s0ppmjpcvprqpdakdu8qqcm2v3z8us334at6";

    // ========================================================================
    // SHA256 Tests
    // ========================================================================

    #[test]
    fn test_sha256_empty() {
        // SHA256("") = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
        let result = sha256::sha256(b"");
        let expected: [u8; 32] = [
            0xe3, 0xb0, 0xc4, 0x42, 0x98, 0xfc, 0x1c, 0x14, 0x9a, 0xfb, 0xf4, 0xc8, 0x99, 0x6f,
            0xb9, 0x24, 0x27, 0xae, 0x41, 0xe4, 0x64, 0x9b, 0x93, 0x4c, 0xa4, 0x95, 0x99, 0x1b,
            0x78, 0x52, 0xb8, 0x55,
        ];
        assert_eq!(result, expected, "SHA256 empty string mismatch");
    }

    #[test]
    fn test_sha256_abc() {
        // SHA256("abc") = ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad
        let result = sha256::sha256(b"abc");
        let expected: [u8; 32] = [
            0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea, 0x41, 0x41, 0x40, 0xde, 0x5d, 0xae,
            0x22, 0x23, 0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c, 0xb4, 0x10, 0xff, 0x61,
            0xf2, 0x00, 0x15, 0xad,
        ];
        assert_eq!(result, expected, "SHA256 'abc' mismatch");
    }

    #[test]
    fn test_sha256_33_bytes() {
        // Test with 33-byte input (compressed pubkey size)
        let result = sha256::sha256(&TEST_PUBKEY_COMPRESSED_1);
        // Expected: SHA256(0279BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798)
        // = 0f715baf5d4c2ed329785cef29e562f73488c8a2bb9dbc5700b361d54b9b0554 (verified with Python)
        let expected: [u8; 32] = [
            0x0f, 0x71, 0x5b, 0xaf, 0x5d, 0x4c, 0x2e, 0xd3, 0x29, 0x78, 0x5c, 0xef, 0x29, 0xe5,
            0x62, 0xf7, 0x34, 0x88, 0xc8, 0xa2, 0xbb, 0x9d, 0xbc, 0x57, 0x00, 0xb3, 0x61, 0xd5,
            0x4b, 0x9b, 0x05, 0x54,
        ];
        assert_eq!(result, expected, "SHA256 pubkey mismatch");
    }

    // ========================================================================
    // RIPEMD160 Tests
    // ========================================================================

    #[test]
    fn test_ripemd160_empty() {
        // RIPEMD160("") = 9c1185a5c5e9fc54612808977ee8f548b2258d31
        let result = ripemd160::ripemd160(b"");
        let expected: [u8; 20] = [
            0x9c, 0x11, 0x85, 0xa5, 0xc5, 0xe9, 0xfc, 0x54, 0x61, 0x28, 0x08, 0x97, 0x7e, 0xe8,
            0xf5, 0x48, 0xb2, 0x25, 0x8d, 0x31,
        ];
        assert_eq!(result, expected, "RIPEMD160 empty string mismatch");
    }

    #[test]
    fn test_ripemd160_abc() {
        // RIPEMD160("abc") = 8eb208f7e05d987a9b044a8e98c6b087f15a0bfc
        let result = ripemd160::ripemd160(b"abc");
        let expected: [u8; 20] = [
            0x8e, 0xb2, 0x08, 0xf7, 0xe0, 0x5d, 0x98, 0x7a, 0x9b, 0x04, 0x4a, 0x8e, 0x98, 0xc6,
            0xb0, 0x87, 0xf1, 0x5a, 0x0b, 0xfc,
        ];
        assert_eq!(result, expected, "RIPEMD160 'abc' mismatch");
    }

    // ========================================================================
    // Hash160 (SHA256 + RIPEMD160) Tests
    // ========================================================================

    #[test]
    fn test_hash160_pubkey_1() {
        // Hash160 of test pubkey 1 should match the address data
        // bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4
        // Decode: w=14, 5=20, 0=15, 8=7, d=13, 6=26, q=0, e=25, j=18, x=6, t=11, d=13, g=8...
        // witness version 'q' = 0
        // Data starts at: w508d6qejxtdg4y5r3zarvary0c5xw7k
        
        let sha256_result = sha256::sha256(&TEST_PUBKEY_COMPRESSED_1);
        let hash160 = ripemd160::ripemd160(&sha256_result);
        
        println!("Hash160 of pubkey 1: {:02x?}", hash160);
        
        // Expected hash160: 751e76e8199196d454941c45d1b3a323f1433bd6
        let expected: [u8; 20] = [
            0x75, 0x1e, 0x76, 0xe8, 0x19, 0x91, 0x96, 0xd4, 0x54, 0x94, 0x1c, 0x45, 0xd1, 0xb3,
            0xa3, 0x23, 0xf1, 0x43, 0x3b, 0xd6,
        ];
        assert_eq!(hash160, expected, "Hash160 pubkey 1 mismatch");
    }

    // ========================================================================
    // Bech32 Encoding Tests
    // ========================================================================

    #[test]
    fn test_bech32_convert_bits() {
        // Test 8-bit to 5-bit conversion
        // hash160: 751e76e8199196d454941c45d1b3a323f1433bd6
        let hash160: [u8; 20] = [
            0x75, 0x1e, 0x76, 0xe8, 0x19, 0x91, 0x96, 0xd4, 0x54, 0x94, 0x1c, 0x45, 0xd1, 0xb3,
            0xa3, 0x23, 0xf1, 0x43, 0x3b, 0xd6,
        ];
        
        let bits5 = bech32::convert_bits_8_to_5(&hash160);
        println!("5-bit conversion: {:?}", bits5);
        
        // Verify length: 160 bits = 32 x 5-bit values
        assert_eq!(bits5.len(), 32, "Expected 32 5-bit values");
        
        // First few values should be:
        // 0x75 = 0111_0101 = [01110, 10100] in 5-bit = [14, 20]
        // 0x1e = 0001_1110 = next bits continue...
        // Actually let's trace through:
        // bytes: 0x75, 0x1e, 0x76, ...
        // bits: 01110101 00011110 01110110 ...
        // 5-bit groups: 01110 10100 01111 00111 0110... = 14, 20, 15, 7, ...
        assert_eq!(bits5[0], 14, "First 5-bit value should be 14 (w)");
        assert_eq!(bits5[1], 20, "Second 5-bit value should be 20 (5)");
        assert_eq!(bits5[2], 15, "Third 5-bit value should be 15 (0)");
    }

    #[test]
    fn test_bech32_encode_address_1() {
        // Full address encoding test
        let hash160: [u8; 20] = [
            0x75, 0x1e, 0x76, 0xe8, 0x19, 0x91, 0x96, 0xd4, 0x54, 0x94, 0x1c, 0x45, 0xd1, 0xb3,
            0xa3, 0x23, 0xf1, 0x43, 0x3b, 0xd6,
        ];
        
        // Prepend witness version 0
        let mut data = vec![0u8]; // witness version 0
        data.extend_from_slice(&bech32::convert_bits_8_to_5(&hash160));
        
        let address = bech32::encode("bc", &data);
        println!("Encoded address: {}", address);
        
        assert_eq!(
            address, TEST_ADDRESS_1,
            "Bech32 encoded address mismatch"
        );
    }

    // ========================================================================
    // Bech32 Character Lookup Tests
    // ========================================================================

    #[test]
    fn test_bech32_charset() {
        // Verify our charset matches Bitcoin's bech32 charset
        const CHARSET: &[u8] = b"qpzry9x8gf2tvdw0s3jn54khce6mua7l";
        
        assert_eq!(CHARSET[0], b'q', "Index 0 should be 'q'");
        assert_eq!(CHARSET[14], b'w', "Index 14 should be 'w'");
        assert_eq!(CHARSET[20], b'5', "Index 20 should be '5'");
        assert_eq!(CHARSET[15], b'0', "Index 15 should be '0'");
        assert_eq!(CHARSET[25], b'e', "Index 25 should be 'e'");
        assert_eq!(CHARSET[12], b'v', "Index 12 should be 'v'");
        assert_eq!(CHARSET[31], b'l', "Index 31 should be 'l'");
        assert_eq!(CHARSET[16], b's', "Index 16 should be 's'");
        assert_eq!(CHARSET[13], b'd', "Index 13 should be 'd'");
    }

    #[test]
    fn test_bech32_reverse_lookup() {
        // Test reverse lookup: char -> 5-bit value
        const BECH32_REV: [i8; 128] = [
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, 15, -1, 10, 17, 21, 20, 26, 30, 7, 5, -1, -1, -1, -1, -1, -1, // 0-9
            -1, 29, -1, 24, 13, 25, 9, 8, 23, -1, 18, 22, 31, 27, 19, -1, // A-O
            1, 0, 3, 16, 11, 28, 12, 14, 6, 4, 2, -1, -1, -1, -1, -1, // P-Z
            -1, 29, -1, 24, 13, 25, 9, 8, 23, -1, 18, 22, 31, 27, 19, -1, // a-o
            1, 0, 3, 16, 11, 28, 12, 14, 6, 4, 2, -1, -1, -1, -1, -1, // p-z
        ];
        
        // Test evlseed pattern
        assert_eq!(BECH32_REV[b'e' as usize], 25, "'e' should map to 25");
        assert_eq!(BECH32_REV[b'v' as usize], 12, "'v' should map to 12");
        assert_eq!(BECH32_REV[b'l' as usize], 31, "'l' should map to 31");
        assert_eq!(BECH32_REV[b's' as usize], 16, "'s' should map to 16");
        // 'e' again = 25
        assert_eq!(BECH32_REV[b'd' as usize], 13, "'d' should map to 13");
        
        // The pattern "evlseed" in 5-bit values should be:
        // e=25, v=12, l=31, s=16, e=25, e=25, d=13
        let expected: Vec<i8> = vec![25, 12, 31, 16, 25, 25, 13];
        let pattern = "evlseed";
        let actual: Vec<i8> = pattern.bytes().map(|c| BECH32_REV[c as usize]).collect();
        assert_eq!(actual, expected, "evlseed 5-bit conversion mismatch");
    }

    // ========================================================================
    // Pattern Matching Tests
    // ========================================================================

    #[test]
    fn test_pattern_parse() {
        let pattern = Pattern::new("bc1qevlseed").unwrap();
        assert_eq!(pattern.hrp, "bc");
        assert_eq!(pattern.data_pattern, "qevlseed");
        
        println!("Pattern: {}", pattern.pattern);
        println!("HRP: {}", pattern.hrp);
        println!("Data pattern: {}", pattern.data_pattern);
        println!("Fast matcher prefix len: {}", pattern.fast_matcher.prefix_len);
    }

    #[test]
    fn test_fast_prefix_matcher() {
        // Test the FastPrefixMatcher with known hash160
        // Pattern: "qevlseed" (data part after bc1)
        // Should skip 'q' (witness version) and match "evlseed"
        
        let matcher = FastPrefixMatcher::new("qevlseed");
        println!("Matcher prefix_5bit: {:?}", matcher.prefix_5bit);
        println!("Matcher prefix_len: {}", matcher.prefix_len);
        
        // Expected: e=25, v=12, l=31, s=16, e=25, e=25, d=13
        let expected_5bit = vec![25u8, 12, 31, 16, 25, 25, 13];
        assert_eq!(
            matcher.prefix_5bit, expected_5bit,
            "FastPrefixMatcher 5-bit conversion mismatch"
        );
        assert_eq!(matcher.prefix_len, 7, "Should have 7 5-bit values");
    }

    #[test]
    fn test_fast_prefix_matcher_matching() {
        // Create a hash160 that starts with "evlseed" when converted to bech32
        // evlseed = [25, 12, 31, 16, 25, 25, 13] in 5-bit
        // = 11001 01100 11111 10000 11001 11001 01101 in binary
        // = 11001_01100_11111_10000_11001_11001_01101
        // Packed into bytes (MSB first):
        // byte0: 11001011 = 0xCB
        // byte1: 00111111 = 0x3F  
        // byte2: 00001100 = 0x0C
        // byte3: 11100101 = 0xE5
        // byte4: 101..... = needs 5 more bits
        
        // Let's compute what hash160 would produce "evlseed"
        // 25=11001, 12=01100, 31=11111, 16=10000, 25=11001, 25=11001, 13=01101
        // bits: 11001 01100 11111 10000 11001 11001 01101
        // grouped by 8: 11001011 00111111 00001100 11100101 101xxxxx
        // hex: CB 3F 0C E5 ...
        
        let mut hash160 = [0u8; 20];
        hash160[0] = 0xCB;
        hash160[1] = 0x3F;
        hash160[2] = 0x0C;
        hash160[3] = 0xE5;
        hash160[4] = 0xA0; // 10100000 - continuing with 101 from byte 4
        
        let matcher = FastPrefixMatcher::new("qevlseed");
        
        println!("Testing hash160: {:02x?}", &hash160[..8]);
        println!("Matcher expects: {:?}", matcher.prefix_5bit);
        
        // Manually convert first bytes to 5-bit to debug
        let mut acc: u32 = 0;
        let mut bits: u32 = 0;
        let mut result_5bit = Vec::new();
        for i in 0..5 {
            acc = (acc << 8) | (hash160[i] as u32);
            bits += 8;
            while bits >= 5 {
                bits -= 5;
                result_5bit.push(((acc >> bits) & 0x1f) as u8);
            }
        }
        println!("Actual 5-bit values from hash: {:?}", result_5bit);
        
        assert!(
            matcher.matches_hash160(&hash160),
            "FastPrefixMatcher should match constructed hash160"
        );
    }

    #[test]
    fn test_fast_prefix_matcher_no_match() {
        // Test with a hash160 that does NOT start with "evlseed"
        let hash160 = [0u8; 20]; // All zeros
        
        let matcher = FastPrefixMatcher::new("qevlseed");
        
        // All zeros in 5-bit would be: 00000 00000 00000 ... = [0, 0, 0, ...]
        // This should NOT match evlseed = [25, 12, 31, ...]
        assert!(
            !matcher.matches_hash160(&hash160),
            "FastPrefixMatcher should NOT match zero hash160"
        );
    }

    // ========================================================================
    // End-to-End Tests
    // ========================================================================

    #[test]
    fn test_pubkey_to_address_1() {
        // Full pipeline: pubkey -> hash160 -> bech32 address
        let sha256_result = sha256::sha256(&TEST_PUBKEY_COMPRESSED_1);
        let hash160 = ripemd160::ripemd160(&sha256_result);
        
        // Convert to bech32
        let mut data = vec![0u8]; // witness version 0
        data.extend_from_slice(&bech32::convert_bits_8_to_5(&hash160));
        let address = bech32::encode("bc", &data);
        
        assert_eq!(address, TEST_ADDRESS_1, "End-to-end address mismatch");
    }

    #[test]
    fn test_decode_found_address() {
        // Decode the address that was "found" to get its hash160
        // bc1qevlseed2qm5j7sfmmcljanzudq0vegeknghzkn
        let decoded = bech32::segwit_decode("bc", TEST_ADDRESS_2);
        assert!(decoded.is_some(), "Failed to decode address");
        
        let (witness_version, witness_program) = decoded.unwrap();
        assert_eq!(witness_version, 0, "Wrong witness version");
        assert_eq!(witness_program.len(), 20, "Wrong witness program length");
        
        println!("Hash160 from address decode: {:02x?}", witness_program);
        
        // Now compute hash160 from the reported private key
        let scalar = Scalar::from_bytes(&TEST_PRIVKEY_2);
        let pubkey = G.mul(&scalar);
        
        let x_bytes = pubkey.x.to_bytes();
        let prefix = if pubkey.y.is_odd() { 0x03 } else { 0x02 };
        let mut compressed = [0u8; 33];
        compressed[0] = prefix;
        compressed[1..].copy_from_slice(&x_bytes);
        
        let sha256_result = sha256::sha256(&compressed);
        let hash160_computed = ripemd160::ripemd160(&sha256_result);
        
        println!("Hash160 from private key: {:02x?}", hash160_computed);
        
        // These should match but currently don't - this shows the bug
        assert_eq!(
            witness_program.as_slice(),
            hash160_computed.as_slice(),
            "Hash160 mismatch - private key doesn't match address!"
        );
    }

    #[test]
    fn test_privkey_to_address() {
        // Full pipeline: private key -> public key -> hash160 -> address
        let scalar = Scalar::from_bytes(&TEST_PRIVKEY_2);
        let pubkey = G.mul(&scalar);
        
        // Compress public key
        let x_bytes = pubkey.x.to_bytes();
        let prefix = if pubkey.y.is_odd() { 0x03 } else { 0x02 };
        let mut compressed = [0u8; 33];
        compressed[0] = prefix;
        compressed[1..].copy_from_slice(&x_bytes);
        
        println!("Public key (compressed): {:02x?}", compressed);
        
        // Hash160
        let sha256_result = sha256::sha256(&compressed);
        let hash160 = ripemd160::ripemd160(&sha256_result);
        println!("Hash160: {:02x?}", hash160);
        
        // Bech32 encode
        let mut data = vec![0u8]; // witness version 0
        data.extend_from_slice(&bech32::convert_bits_8_to_5(&hash160));
        let address = bech32::encode("bc", &data);
        println!("Address: {}", address);
        
        assert_eq!(address, TEST_ADDRESS_2, "Private key -> address mismatch");
    }

    // ========================================================================
    // GPU vs CPU Consistency Tests
    // ========================================================================

    #[test]
    fn test_hash160_byte_order() {
        // This test verifies the byte order of hash160 
        // The GPU uses little-endian uint32 internally
        
        let sha256_result = sha256::sha256(&TEST_PUBKEY_COMPRESSED_1);
        let hash160 = ripemd160::ripemd160(&sha256_result);
        
        // Convert to 5 x uint32 (little-endian, as GPU would store)
        let mut h = [0u32; 5];
        for i in 0..5 {
            h[i] = u32::from_le_bytes([
                hash160[i * 4],
                hash160[i * 4 + 1],
                hash160[i * 4 + 2],
                hash160[i * 4 + 3],
            ]);
        }
        println!("Hash160 as LE u32: {:08x?}", h);
        
        // Now extract bytes the way the GPU does in _CheckBech32Pattern:
        // bytes[0] = (h[0] >> 0) & 0xFF = hash160[0]
        // bytes[1] = (h[0] >> 8) & 0xFF = hash160[1]
        // bytes[2] = (h[0] >> 16) & 0xFF = hash160[2]
        // bytes[3] = (h[0] >> 24) & 0xFF = hash160[3]
        
        let b0 = (h[0] >> 0) & 0xFF;
        let b1 = (h[0] >> 8) & 0xFF;
        let b2 = (h[0] >> 16) & 0xFF;
        let b3 = (h[0] >> 24) & 0xFF;
        
        println!("GPU byte extraction: {:02x} {:02x} {:02x} {:02x}", b0, b1, b2, b3);
        println!("Expected (hash160):  {:02x} {:02x} {:02x} {:02x}", 
                 hash160[0], hash160[1], hash160[2], hash160[3]);
        
        assert_eq!(b0 as u8, hash160[0], "Byte 0 mismatch");
        assert_eq!(b1 as u8, hash160[1], "Byte 1 mismatch");
        assert_eq!(b2 as u8, hash160[2], "Byte 2 mismatch");
        assert_eq!(b3 as u8, hash160[3], "Byte 3 mismatch");
        
        // Then GPU creates data32 = (b0 << 24) | (b1 << 16) | (b2 << 8) | b3
        // This is big-endian reordering
        let data32 = ((b0 as u32) << 24) | ((b1 as u32) << 16) | ((b2 as u32) << 8) | (b3 as u32);
        println!("GPU data32: {:08x}", data32);
        
        // First 5-bit value: (data32 >> 27) & 0x1F
        let first_5bit = (data32 >> 27) & 0x1F;
        println!("GPU first 5-bit: {} (expected: {})", first_5bit, hash160[0] >> 3);
        
        // CPU extracts: hash160[0] >> 3 for first 5 bits
        let cpu_first_5bit = hash160[0] >> 3;
        assert_eq!(first_5bit, cpu_first_5bit as u32, "First 5-bit value mismatch");
    }

    #[test]
    fn test_5bit_conversion_cpu_vs_manual() {
        // Verify CPU 5-bit conversion matches manual calculation
        let hash160: [u8; 20] = [
            0x75, 0x1e, 0x76, 0xe8, 0x19, 0x91, 0x96, 0xd4, 0x54, 0x94, 0x1c, 0x45, 0xd1, 0xb3,
            0xa3, 0x23, 0xf1, 0x43, 0x3b, 0xd6,
        ];
        
        // Manual calculation for first 6 values:
        // bytes: 0x75 0x1e 0x76 0xe8
        // binary: 01110101 00011110 01110110 11101000
        // 5-bit groups: 01110 10100 01111 00111 01101 11010...
        // decimal: 14 20 15 7 13 26...
        let expected_first_6 = [14u8, 20, 15, 7, 13, 26];
        
        let bits5 = bech32::convert_bits_8_to_5(&hash160);
        
        println!("First 6 5-bit values: {:?}", &bits5[..6]);
        println!("Expected: {:?}", expected_first_6);
        
        for i in 0..6 {
            assert_eq!(bits5[i], expected_first_6[i], "5-bit value {} mismatch", i);
        }
    }

    // ========================================================================
    // GPU Pattern Matching Simulation
    // ========================================================================

    #[test]
    fn test_gpu_pattern_check_simulation() {
        // Simulate the GPU's _CheckBech32Pattern function
        // This is critical for understanding if GPU and CPU match
        
        // Test with known hash160 from address bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4
        let hash160: [u8; 20] = [
            0x75, 0x1e, 0x76, 0xe8, 0x19, 0x91, 0x96, 0xd4, 0x54, 0x94, 0x1c, 0x45, 0xd1, 0xb3,
            0xa3, 0x23, 0xf1, 0x43, 0x3b, 0xd6,
        ];
        
        // Pattern: w508d6 (first 6 chars after 'q')
        // w=14, 5=20, 0=15, 8=7, d=13, 6=26
        let pattern5bit: [u8; 6] = [14, 20, 15, 7, 13, 26];
        
        // Simulate GPU byte extraction (as in _CheckBech32Pattern)
        let h = [
            u32::from_le_bytes([hash160[0], hash160[1], hash160[2], hash160[3]]),
            u32::from_le_bytes([hash160[4], hash160[5], hash160[6], hash160[7]]),
            u32::from_le_bytes([hash160[8], hash160[9], hash160[10], hash160[11]]),
            u32::from_le_bytes([hash160[12], hash160[13], hash160[14], hash160[15]]),
            u32::from_le_bytes([hash160[16], hash160[17], hash160[18], hash160[19]]),
        ];
        
        let h0 = h[0];
        let b0 = (h0 >> 0) & 0xFF;
        let b1 = (h0 >> 8) & 0xFF;
        let b2 = (h0 >> 16) & 0xFF;
        let b3 = (h0 >> 24) & 0xFF;
        
        // GPU creates data32 in big-endian
        let data32 = ((b0 as u32) << 24) | ((b1 as u32) << 16) | ((b2 as u32) << 8) | (b3 as u32);
        
        println!("h[0] = 0x{:08x}", h0);
        println!("bytes: {:02x} {:02x} {:02x} {:02x}", b0, b1, b2, b3);
        println!("data32 = 0x{:08x}", data32);
        
        // Check each 5-bit value
        let val0 = (data32 >> 27) & 0x1F;
        let val1 = (data32 >> 22) & 0x1F;
        let val2 = (data32 >> 17) & 0x1F;
        let val3 = (data32 >> 12) & 0x1F;
        let val4 = (data32 >> 7) & 0x1F;
        let val5 = (data32 >> 2) & 0x1F;
        
        println!("GPU 5-bit values: {} {} {} {} {} {}", val0, val1, val2, val3, val4, val5);
        println!("Expected pattern: {:?}", pattern5bit);
        
        assert_eq!(val0, pattern5bit[0] as u32, "5-bit[0] mismatch");
        assert_eq!(val1, pattern5bit[1] as u32, "5-bit[1] mismatch");
        assert_eq!(val2, pattern5bit[2] as u32, "5-bit[2] mismatch");
        assert_eq!(val3, pattern5bit[3] as u32, "5-bit[3] mismatch");
        assert_eq!(val4, pattern5bit[4] as u32, "5-bit[4] mismatch");
        assert_eq!(val5, pattern5bit[5] as u32, "5-bit[5] mismatch");
    }

    // ========================================================================
    // Difficulty Calculation Test
    // ========================================================================

    #[test]
    fn test_difficulty_calculation() {
        // Pattern "evlseed" = 7 characters = 35 bits
        // Expected: 2^35 = 34,359,738,368 keys per match
        
        let pattern = Pattern::new("bc1qevlseed").unwrap();
        let expected_difficulty = 2.0_f64.powi(35);
        
        println!("Pattern difficulty: {}", pattern.difficulty);
        println!("Expected (2^35): {}", expected_difficulty);
        
        // Allow 1% tolerance for floating point
        let ratio = pattern.difficulty / expected_difficulty;
        assert!(
            (0.99..=1.01).contains(&ratio),
            "Difficulty calculation off: got {}, expected {}",
            pattern.difficulty,
            expected_difficulty
        );
    }

    // ========================================================================
    // GPU Hash160 Byte Order Test
    // ========================================================================

    #[test]
    fn test_ripemd160_output_format() {
        // Test that RIPEMD160 output is in the expected byte order
        // for the GPU's little-endian uint32 storage
        
        let sha256_result = sha256::sha256(&TEST_PUBKEY_COMPRESSED_1);
        let hash160 = ripemd160::ripemd160(&sha256_result);
        
        println!("Hash160 bytes: {:02x?}", hash160);
        
        // When stored as little-endian uint32:
        // h[0] = bytes[0] | (bytes[1] << 8) | (bytes[2] << 16) | (bytes[3] << 24)
        let h0 = (hash160[0] as u32)
            | ((hash160[1] as u32) << 8)
            | ((hash160[2] as u32) << 16)
            | ((hash160[3] as u32) << 24);
        
        println!("h[0] as LE u32: 0x{:08x}", h0);
        
        // Verify we can extract bytes back correctly
        assert_eq!((h0 >> 0) as u8, hash160[0]);
        assert_eq!((h0 >> 8) as u8, hash160[1]);
        assert_eq!((h0 >> 16) as u8, hash160[2]);
        assert_eq!((h0 >> 24) as u8, hash160[3]);
    }

    // ========================================================================
    // GPU Key Reconstruction Tests
    // ========================================================================

    /// Test the GPU key reconstruction formula:
    /// Given a base_key and incr value from GPU, reconstruct the private key
    /// 
    /// GPU stores: incr relative to center (GRP_SIZE/2 = 512)
    /// - For incr >= 0: point is at base_key + (incr - 512)
    /// - For incr < 0: point is negated, scalar = -(base_key + (|incr| - 512))
    #[test]
    fn test_gpu_key_reconstruction_center() {
        // Test case: center of group (incr = 512)
        // The key should be exactly base_key
        const GRP_SIZE_HALF: i32 = 512;
        
        let base_key = Scalar::from_bytes(&TEST_PRIVKEY_2);
        let incr: i32 = GRP_SIZE_HALF; // 512 = center
        let endo: i32 = 0;
        
        // Compute reconstructed key
        let incr_abs = incr.abs();
        let offset = incr_abs - GRP_SIZE_HALF;  // = 0
        
        // When offset is 0, key should equal base_key
        let reconstructed = if offset == 0 {
            base_key
        } else if offset > 0 {
            let mut offset_bytes = [0u8; 32];
            offset_bytes[31] = offset as u8;
            base_key.add(&Scalar::from_bytes(&offset_bytes))
        } else {
            let mut offset_bytes = [0u8; 32];
            offset_bytes[31] = (-offset) as u8;
            base_key.sub(&Scalar::from_bytes(&offset_bytes))
        };
        
        // Verify
        assert_eq!(reconstructed.to_bytes(), base_key.to_bytes(), 
            "Center key (incr=512) should equal base_key");
    }
    
    #[test]
    fn test_gpu_key_reconstruction_positive_offset() {
        // Test case: incr = 513 -> offset = 1 -> key = base_key + 1
        const GRP_SIZE_HALF: i32 = 512;
        
        let base_key = Scalar::from_bytes(&TEST_PRIVKEY_2);
        let incr: i32 = 513;
        let offset = incr - GRP_SIZE_HALF;  // = 1
        
        // Manually compute expected: base_key + 1
        let mut one = [0u8; 32];
        one[31] = 1;
        let expected = base_key.add(&Scalar::from_bytes(&one));
        
        // Reconstruct using formula
        let mut offset_bytes = [0u8; 32];
        offset_bytes[31] = offset as u8;
        let reconstructed = base_key.add(&Scalar::from_bytes(&offset_bytes));
        
        assert_eq!(reconstructed.to_bytes(), expected.to_bytes(),
            "incr=513 should give base_key + 1");
        
        // Verify the reconstructed key produces a valid address
        let pubkey = G.mul(&reconstructed);
        let x_bytes = pubkey.x.to_bytes();
        let prefix = if pubkey.y.is_odd() { 0x03 } else { 0x02 };
        let mut compressed = [0u8; 33];
        compressed[0] = prefix;
        compressed[1..].copy_from_slice(&x_bytes);
        
        let sha256_result = sha256::sha256(&compressed);
        let hash160 = ripemd160::ripemd160(&sha256_result);
        let mut data = vec![0u8];
        data.extend_from_slice(&bech32::convert_bits_8_to_5(&hash160));
        let address = bech32::encode("bc", &data);
        
        println!("Reconstructed key: {:?}", reconstructed.to_bytes());
        println!("Derived address: {}", address);
        assert!(address.starts_with("bc1q"), "Should produce valid bech32 address");
    }
    
    #[test]
    fn test_gpu_key_reconstruction_negative_incr() {
        // Test case: negative incr means the point was negated on GPU
        // incr = -512 -> incr_abs = 512 -> offset = 0 -> key = -(base_key)
        const GRP_SIZE_HALF: i32 = 512;
        
        let base_key = Scalar::from_bytes(&TEST_PRIVKEY_2);
        let incr: i32 = -512;
        
        // With negative incr, we negate the final key
        let incr_abs = incr.abs();
        let offset = incr_abs - GRP_SIZE_HALF;  // = 0
        
        let mut reconstructed = base_key;
        if offset > 0 {
            let mut offset_bytes = [0u8; 32];
            offset_bytes[31] = offset as u8;
            reconstructed = reconstructed.add(&Scalar::from_bytes(&offset_bytes));
        }
        // Negate because incr was negative
        reconstructed = reconstructed.neg();
        
        // Verify: -base_key should produce the same x-coordinate but negated y
        let pub_original = G.mul(&base_key);
        let pub_negated = G.mul(&reconstructed);
        
        assert_eq!(pub_original.x.to_bytes(), pub_negated.x.to_bytes(),
            "Negated key should have same x-coordinate");
    }
    
    #[test]
    fn test_gpu_key_reconstruction_with_endomorphism() {
        // Test case: endo = 1 means multiply key by lambda1
        // For endo=1: k' = k * lambda1 mod n
        
        let base_key = Scalar::from_bytes(&TEST_PRIVKEY_2);
        
        // lambda1 = 0x5363ad4cc05c30e0a5261c028812645a122e22ea20816678df02967c1b23bd72
        let lambda1_bytes: [u8; 32] = [
            0x53, 0x63, 0xad, 0x4c, 0xc0, 0x5c, 0x30, 0xe0, 0xa5, 0x26, 0x1c, 0x02,
            0x88, 0x12, 0x64, 0x5a, 0x12, 0x2e, 0x22, 0xea, 0x20, 0x81, 0x66, 0x78,
            0xdf, 0x02, 0x96, 0x7c, 0x1b, 0x23, 0xbd, 0x72,
        ];
        let lambda1 = Scalar::from_bytes(&lambda1_bytes);
        
        let reconstructed = base_key.mul(&lambda1);
        
        // The reconstructed key should produce a valid point
        let pubkey = G.mul(&reconstructed);
        println!("Endomorphism key pubkey.x: {:?}", pubkey.x.to_bytes());
        
        // Verify this relates to the beta-multiplied x-coordinate
        // For endo=1, the GPU computes: x' = beta * x (mod p)
        // where beta = 0x7ae96a2b657c07106e64479eac3434e99cf0497512f58995c1396c28719501ee
        let original_pub = G.mul(&base_key);
        
        // Beta in field
        let beta_bytes: [u8; 32] = [
            0x7a, 0xe9, 0x6a, 0x2b, 0x65, 0x7c, 0x07, 0x10, 0x6e, 0x64, 0x47, 0x9e,
            0xac, 0x34, 0x34, 0xe9, 0x9c, 0xf0, 0x49, 0x75, 0x12, 0xf5, 0x89, 0x95,
            0xc1, 0x39, 0x6c, 0x28, 0x71, 0x95, 0x01, 0xee,
        ];
        use crate::secp256k1::FieldElement;
        let beta = FieldElement::from_bytes(&beta_bytes);
        let expected_x = original_pub.x.mul(&beta);
        
        // pubkey.x should equal beta * original_pub.x
        assert_eq!(pubkey.x.to_bytes(), expected_x.to_bytes(),
            "Endomorphism should produce x' = beta * x");
    }

    #[test]
    fn test_gpu_output_parsing() {
        // Test parsing GPU output format
        // GPU stores: out[pos * 8 + 1] = tid, out[pos * 8 + 2] = info, etc.
        // info = (incr << 16) | endo
        
        // Simulate GPU output
        let info: u32 = (512u32 << 16) | 0;  // incr=512, endo=0
        let parsed_incr = ((info >> 16) as i16) as i32;
        let parsed_endo = (info & 0x7) as i32;
        
        assert_eq!(parsed_incr, 512, "Should parse incr correctly");
        assert_eq!(parsed_endo, 0, "Should parse endo correctly");
        
        // Test negative incr
        let neg_incr: i16 = -512;
        let info_neg: u32 = ((neg_incr as u16 as u32) << 16) | 1;  // incr=-512, endo=1
        let parsed_neg_incr = ((info_neg >> 16) as i16) as i32;
        let parsed_neg_endo = (info_neg & 0x7) as i32;
        
        assert_eq!(parsed_neg_incr, -512, "Should parse negative incr correctly");
        assert_eq!(parsed_neg_endo, 1, "Should parse endo with negative incr");
    }

    #[test]
    fn test_end_to_end_key_reconstruction() {
        // Complete end-to-end test:
        // 1. Start with known base_key
        // 2. Simulate GPU finding a match at specific incr/endo
        // 3. Reconstruct the key
        // 4. Verify the reconstructed key produces expected address
        
        const GRP_SIZE_HALF: i32 = 512;
        
        // Known base key
        let base_key = Scalar::from_bytes(&TEST_PRIVKEY_2);
        
        // Test various incr values
        let test_cases: [(i32, i32, &str); 3] = [
            (512, 0, "center"),      // Center, no endo
            (513, 0, "offset +1"),   // Offset +1, no endo
            (511, 0, "offset -1"),   // Offset -1, no endo
        ];
        
        for (incr, endo, description) in test_cases {
            let offset = incr - GRP_SIZE_HALF;
            
            // Reconstruct key (same logic as process_matches)
            let mut private_key = base_key;
            
            if offset > 0 {
                let mut offset_bytes = [0u8; 32];
                offset_bytes[31] = offset as u8;
                private_key = private_key.add(&Scalar::from_bytes(&offset_bytes));
            } else if offset < 0 {
                let mut offset_bytes = [0u8; 32];
                offset_bytes[31] = (-offset) as u8;
                private_key = private_key.sub(&Scalar::from_bytes(&offset_bytes));
            }
            
            // No endomorphism for endo=0
            // No negation for positive incr
            
            // Derive address
            let pubkey = G.mul(&private_key);
            let x_bytes = pubkey.x.to_bytes();
            let prefix = if pubkey.y.is_odd() { 0x03 } else { 0x02 };
            let mut compressed = [0u8; 33];
            compressed[0] = prefix;
            compressed[1..].copy_from_slice(&x_bytes);
            
            let sha256_result = sha256::sha256(&compressed);
            let hash160 = ripemd160::ripemd160(&sha256_result);
            let mut data = vec![0u8];
            data.extend_from_slice(&bech32::convert_bits_8_to_5(&hash160));
            let address = bech32::encode("bc", &data);
            
            println!("{}: incr={}, offset={}, address={}", description, incr, offset, address);
            
            // The address should be a valid bech32 address
            assert!(address.starts_with("bc1q"), 
                "{}: Should produce valid address", description);
            
            // For center (offset=0), should match TEST_ADDRESS_2
            if incr == GRP_SIZE_HALF {
                assert_eq!(address, TEST_ADDRESS_2,
                    "Center key should produce known test address");
            }
        }
    }
}

