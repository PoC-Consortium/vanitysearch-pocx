use pocx_vanity::crypto::{self, Crypto, bech32_utils};

#[test]
fn test_hash160() {
    // Known test vector
    let pubkey = hex::decode("0279BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798").unwrap();
    let hash = crypto::hash160(&pubkey);
    
    // Bitcoin's hash160 of G point compressed
    let expected = hex::decode("751e76e8199196d454941c45d1b3a323f1433bd6").unwrap();
    assert_eq!(hash, expected.as_slice());
}

#[test]
fn test_bech32_encode_decode() {
    let hash160: [u8; 20] = hex::decode("751e76e8199196d454941c45d1b3a323f1433bd6")
        .unwrap()
        .try_into()
        .unwrap();
    
    let address = bech32_utils::encode(&hash160, true);
    assert!(address.starts_with("pocx1q"));
    
    let decoded = bech32_utils::decode(&address).unwrap();
    assert_eq!(hash160, decoded);
}

#[test]
fn test_bech32_prefix_decode() {
    let data = bech32_utils::decode_prefix("pocx1qma").unwrap();
    assert_eq!(data.len(), 2); // "ma" = 2 chars after q
}

#[test]
fn test_difficulty_calculation() {
    assert_eq!(bech32_utils::difficulty("pocx1qa"), 32.0);      // 2^5
    assert_eq!(bech32_utils::difficulty("pocx1qaa"), 1024.0);   // 2^10
    assert_eq!(bech32_utils::difficulty("pocx1qaaa"), 32768.0); // 2^15
}

#[test]
fn test_wif_generation() {
    let crypto = Crypto::new();
    
    // Test with known private key
    let priv_key: [u8; 32] = hex::decode(
        "2233181AC0DA99DC48737C256EE44DC6FAF3FF1C9AE3EC4A42053540B0EF7EBD"
    ).unwrap().try_into().unwrap();
    
    let wif = crypto.to_wif(&priv_key, true);
    assert_eq!(wif, "KxNBzneS1F2AmRacLWfHa3XcoCPtKiZCtwEreW9kio4b9ewe5HzN");
}
