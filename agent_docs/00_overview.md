# PoCX Vanity Search - Rust Implementation

## Ziel

GPU-beschleunigter Vanity Address Generator für `pocx1q...` Bech32 Adressen in Rust mit:
- **CUDA** für NVIDIA GPUs (primär, höchste Performance)
- **OpenCL** als Fallback (NVIDIA, AMD, Intel)

## Scope

- Nur `pocx1q` Bech32 (HRP="pocx", witness v0)
- Nur komprimierte Public Keys (33 Bytes)
- Lowercase only
- Ausgabe: Mainnet + Testnet Adressen/WIFs

## Architektur

```
???????????????????????????????????????????
?              CLI (clap)                 ?
???????????????????????????????????????????
?           VanitySearch Core             ?
???????????????????????????????????????????
?   CUDA Backend   ?   OpenCL Backend     ?
?   (cudarc/cu.rs) ?   (opencl3/gpgpu)    ?
???????????????????????????????????????????
?         Crypto (CPU Verification)       ?
?    secp256k1, sha2, ripemd160, bech32   ?
???????????????????????????????????????????
```

## Projektstruktur

```
pocx-vanity/
??? Cargo.toml
??? src/
?   ??? main.rs
?   ??? lib.rs
?   ??? config.rs
?   ??? result.rs
?   ??? search.rs
?   ??? crypto/
?   ?   ??? mod.rs
?   ?   ??? secp256k1.rs
?   ?   ??? bech32.rs
?   ??? gpu/
?   ?   ??? mod.rs
?   ?   ??? detect.rs
?   ?   ??? cuda/
?   ?   ?   ??? mod.rs
?   ?   ?   ??? context.rs
?   ?   ?   ??? kernels/
?   ?   ?       ??? vanity.cu
?   ?   ??? opencl/
?   ?       ??? mod.rs
?   ?       ??? context.rs
?   ?       ??? kernels/
?   ?           ??? vanity.cl
?   ??? output.rs
??? tests/
?   ??? crypto_tests.rs
?   ??? cuda_opencl_parity.rs
?   ??? known_vectors.rs
??? benches/
    ??? throughput.rs
```

## Dependencies (Cargo.toml)

```toml
[package]
name = "pocx-vanity"
version = "0.1.0"
edition = "2021"

[dependencies]
# CLI
clap = { version = "4", features = ["derive"] }

# Crypto
secp256k1 = { version = "0.28", features = ["rand"] }
sha2 = "0.10"
ripemd = "0.1"
bech32 = "0.9"
bs58 = "0.5"
hex = "0.4"
rand = "0.8"

# GPU - CUDA
cudarc = { version = "0.10", optional = true }

# GPU - OpenCL
opencl3 = "0.9"

# Async
tokio = { version = "1", features = ["full"] }

# Utility
thiserror = "1"
anyhow = "1"
bytemuck = { version = "1", features = ["derive"] }

[features]
default = ["cuda", "opencl"]
cuda = ["cudarc"]
opencl = []

[dev-dependencies]
criterion = "0.5"

[[bench]]
name = "throughput"
harness = false
```

## GPU-Erkennung Logik

```rust
pub enum GpuBackend {
    Cuda(CudaDevice),
    OpenCL(OpenCLDevice),
}

pub fn detect_best_gpu() -> Result<GpuBackend> {
    // 1. Versuche CUDA (nur NVIDIA)
    if cfg!(feature = "cuda") {
        if let Ok(device) = CudaContext::detect() {
            return Ok(GpuBackend::Cuda(device));
        }
    }
    
    // 2. Fallback: OpenCL
    if let Ok(device) = OpenCLContext::detect_gpu() {
        return Ok(GpuBackend::OpenCL(device));
    }
    
    Err(anyhow!("No GPU found"))
}
```

## Test-Vektoren (verifiziert)

```rust
const TEST_VECTORS: &[(&str, &str, &str)] = &[
    // (private_key_hex, address, wif)
    ("2233181AC0DA99DC48737C256EE44DC6FAF3FF1C9AE3EC4A42053540B0EF7EBD",
     "pocx1qmadf0xkl7qllnq06cprxehqeeenkn7hg26pgrf",
     "KxNBzneS1F2AmRacLWfHa3XcoCPtKiZCtwEreW9kio4b9ewe5HzN"),
    ("689FDD5BFAEB3F4D0B01DA7B2EFA5554C504190389CE0E85701DAACFF4A18146",
     "pocx1qmadf0xrcf4ljapsu23ppdcmammfet4jw555sk5",
     "Kzj61Xuk6hra69TJWLPvapoKZMWb71vnQELbzCKEkAjh5zL78qFK"),
    ("3932C5058E39F5A5753C1947B49EAF8D7CE937098721D642CCFDAD2CC2847671",
     "pocx1qmadf0x5ulmn5vzv76c933hpgjjw5zwgvruqddw",
     "Ky8tyZADEtBmx5wnArSETqnomGgx8Mg9pwCMSBcNB2oSmvs2urGS"),
];
```
