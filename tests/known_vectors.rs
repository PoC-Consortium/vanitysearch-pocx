use pocx_vanity::crypto::{self, Crypto, bech32_utils};

/// Test vectors from working CUDA implementation
const VECTORS: &[(&str, &str, &str)] = &[
    (
        "2233181AC0DA99DC48737C256EE44DC6FAF3FF1C9AE3EC4A42053540B0EF7EBD",
        "pocx1qmadf0xkl7qllnq06cprxehqeeenkn7hg26pgrf",
        "KxNBzneS1F2AmRacLWfHa3XcoCPtKiZCtwEreW9kio4b9ewe5HzN"
    ),
    (
        "689FDD5BFAEB3F4D0B01DA7B2EFA5554C504190389CE0E85701DAACFF4A18146",
        "pocx1qmadf0xrcf4ljapsu23ppdcmammfet4jw555sk5",
        "Kzj61Xuk6hra69TJWLPvapoKZMWb71vnQELbzCKEkAjh5zL78qFK"
    ),
    (
        "3932C5058E39F5A5753C1947B49EAF8D7CE937098721D642CCFDAD2CC2847671",
        "pocx1qmadf0x5ulmn5vzv76c933hpgjjw5zwgvruqddw",
        "Ky8tyZADEtBmx5wnArSETqnomGgx8Mg9pwCMSBcNB2oSmvs2urGS"
    ),
    (
        "D1B2D2BE6A711C3372D5A4A056B162E5A66EB6F37925E8D6CAB0997901940257",
        "pocx1qmadf0xljv3p5kplm04fsv2ygcjm9hfc0ytma8c",
        "L4FLXKw7hV4DurPyKFwFDa886jtsMsVqQDK94zZxjvF8oXcg29YM"
    ),
    (
        "2C1ADF939E2A61B6F679C6205F488E9B2FCA836DEC5F657CBA115DB4B624440E",
        "pocx1qmadf0x267lmwfqndhj4l2ymtsvhvxe5vptnsj5",
        "KxhSms42xBNMMkFjVxxVrVUJvMkD7igKyc1TN7GKH6K1r9TEZqBm"
    ),
];

#[test]
fn test_private_key_to_address() {
    let crypto = Crypto::new();
    
    for (priv_hex, expected_addr, expected_wif) in VECTORS {
        let priv_key: [u8; 32] = hex::decode(priv_hex)
            .unwrap()
            .try_into()
            .unwrap();
        
        // Generate public key
        let pubkey = crypto.public_key(&priv_key).unwrap();
        
        // Hash160
        let hash160 = crypto::hash160(&pubkey);
        
        // Bech32 encode
        let address = bech32_utils::encode(&hash160, true);
        
        assert_eq!(&address, *expected_addr, "Address mismatch for key {}", priv_hex);
        
        // WIF
        let wif = crypto.to_wif(&priv_key, true);
        assert_eq!(&wif, *expected_wif, "WIF mismatch for key {}", priv_hex);
    }
}

#[test]
fn test_all_addresses_start_with_madf0x() {
    // All test vectors have "madf0x" prefix
    for (_, addr, _) in VECTORS {
        assert!(addr.starts_with("pocx1qmadf0x"), "Expected prefix 'pocx1qmadf0x' in {}", addr);
    }
}

#[test]
fn test_testnet_address() {
    let crypto = Crypto::new();
    
    let priv_key: [u8; 32] = hex::decode(
        "2233181AC0DA99DC48737C256EE44DC6FAF3FF1C9AE3EC4A42053540B0EF7EBD"
    ).unwrap().try_into().unwrap();
    
    let pubkey = crypto.public_key(&priv_key).unwrap();
    let hash160 = crypto::hash160(&pubkey);
    
    let address = bech32_utils::encode(&hash160, false);
    assert!(address.starts_with("tpocx1q"), "Testnet address should start with 'tpocx1q'");
}

#[test]
fn test_testnet_wif() {
    let crypto = Crypto::new();
    
    let priv_key: [u8; 32] = hex::decode(
        "2233181AC0DA99DC48737C256EE44DC6FAF3FF1C9AE3EC4A42053540B0EF7EBD"
    ).unwrap().try_into().unwrap();
    
    let wif_mainnet = crypto.to_wif(&priv_key, true);
    let wif_testnet = crypto.to_wif(&priv_key, false);
    
    // Mainnet WIF starts with K or L (compressed)
    assert!(wif_mainnet.starts_with('K') || wif_mainnet.starts_with('L'));
    
    // Testnet WIF starts with c (compressed)
    assert!(wif_testnet.starts_with('c'));
}
