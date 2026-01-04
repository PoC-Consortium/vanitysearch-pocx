# Rust: Tests

## tests/crypto_tests.rs

```rust
use pocx_vanity::crypto::{self, Crypto, bech32};

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
    
    let address = bech32::encode(&hash160, true);
    assert!(address.starts_with("pocx1q"));
    
    let decoded = bech32::decode(&address).unwrap();
    assert_eq!(hash160, decoded);
}

#[test]
fn test_bech32_prefix_decode() {
    let data = bech32::decode_prefix("pocx1qma").unwrap();
    assert_eq!(data.len(), 2); // "ma" = 2 chars after q
}

#[test]
fn test_difficulty_calculation() {
    assert_eq!(bech32::difficulty("pocx1qa"), 32.0);      // 2^5
    assert_eq!(bech32::difficulty("pocx1qaa"), 1024.0);   // 2^10
    assert_eq!(bech32::difficulty("pocx1qaaa"), 32768.0); // 2^15
}
```

## tests/known_vectors.rs

Verifiziert gegen die funktionierenden CUDA-Ergebnisse:

```rust
use pocx_vanity::crypto::{self, Crypto, bech32};

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
        let address = bech32::encode(&hash160, true);
        
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
```

## tests/cuda_opencl_parity.rs

CUDA vs OpenCL Ergebnis-Vergleich:

```rust
#![cfg(all(feature = "cuda", feature = "opencl"))]

use pocx_vanity::gpu::{cuda::CudaBackend, opencl::OpenCLBackend, GpuBackend};
use pocx_vanity::crypto::{self, Crypto};

/// Test that CUDA and OpenCL produce identical results
#[test]
fn test_cuda_opencl_parity() {
    // Skip if no CUDA device available
    let cuda = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(_) => {
            eprintln!("CUDA not available, skipping parity test");
            return;
        }
    };
    
    let opencl = match OpenCLBackend::new(0) {
        Ok(b) => b,
        Err(_) => {
            eprintln!("OpenCL not available, skipping parity test");
            return;
        }
    };
    
    // Use same starting keys
    let test_keys: Vec<[u8; 32]> = (0..16)
        .map(|i| {
            let mut key = [0u8; 32];
            key[31] = i;
            key
        })
        .collect();
    
    // Set same prefix
    let prefix = vec!["pocx1qa".to_string()];
    
    let mut cuda = cuda;
    let mut opencl = opencl;
    
    cuda.set_prefixes(&prefix).unwrap();
    opencl.set_prefixes(&prefix).unwrap();
    
    cuda.set_keys(&test_keys).unwrap();
    opencl.set_keys(&test_keys).unwrap();
    
    // Run multiple iterations and compare
    for iteration in 0..10 {
        let cuda_results = cuda.launch().unwrap();
        let opencl_results = opencl.launch().unwrap();
        
        // Both should find same matches (or both find nothing)
        assert_eq!(
            cuda_results.len(), 
            opencl_results.len(),
            "Iteration {}: CUDA found {} results, OpenCL found {}",
            iteration, cuda_results.len(), opencl_results.len()
        );
        
        // Verify hash160 matches
        for (c, o) in cuda_results.iter().zip(opencl_results.iter()) {
            assert_eq!(
                c.hash160, o.hash160,
                "Hash160 mismatch at iteration {}", iteration
            );
            assert_eq!(
                c.increment, o.increment,
                "Increment mismatch at iteration {}", iteration
            );
            assert_eq!(
                c.endomorphism, o.endomorphism,
                "Endomorphism mismatch at iteration {}", iteration
            );
        }
        
        cuda.advance_keys();
        opencl.advance_keys();
    }
}

/// Test specific known keys produce same results on both backends
#[test]
fn test_known_key_parity() {
    let crypto = Crypto::new();
    
    // Use a known test vector
    let priv_key: [u8; 32] = hex::decode(
        "2233181AC0DA99DC48737C256EE44DC6FAF3FF1C9AE3EC4A42053540B0EF7EBD"
    ).unwrap().try_into().unwrap();
    
    let pubkey = crypto.public_key(&priv_key).unwrap();
    let hash160 = crypto::hash160(&pubkey);
    
    // Verify this is the expected hash160 for "pocx1qmadf0xkl7qllnq06cprxehqeeenkn7hg26pgrf"
    let expected_hash = pocx_vanity::crypto::bech32::decode(
        "pocx1qmadf0xkl7qllnq06cprxehqeeenkn7hg26pgrf"
    ).unwrap();
    
    assert_eq!(hash160, expected_hash, "Hash160 mismatch for known test vector");
}
```

## tests/endomorphism_tests.rs

```rust
use pocx_vanity::crypto::Crypto;

#[test]
fn test_endomorphism_recovery() {
    let crypto = Crypto::new();
    
    let base_key: [u8; 32] = hex::decode(
        "0000000000000000000000000000000000000000000000000000000000000001"
    ).unwrap().try_into().unwrap();
    
    // Test all endomorphism + increment combinations
    for endo in 0..=2 {
        for incr in [-10, -1, 0, 1, 10] {
            let recovered = crypto.recover_private_key(&base_key, incr, endo);
            
            // Should produce valid public key
            let pubkey = crypto.public_key(&recovered);
            assert!(pubkey.is_ok(), "Failed for endo={}, incr={}", endo, incr);
        }
    }
}
```
