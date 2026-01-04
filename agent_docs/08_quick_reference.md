# Rust: Quick Reference

## Secp256k1 Konstanten (Little Endian u64[4])

```rust
const P: [u64; 4] = [0xFFFFFFFFFFFFFC2F, 0xFFFFFFFFFFFFFFFE, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF];
const N: [u64; 4] = [0xBFD25E8CD0364141, 0xBAAEDCE6AF48A03B, 0xFFFFFFFFFFFFFFFE, 0xFFFFFFFFFFFFFFFF];
const GX: [u64; 4] = [0x59F2815B16F81798, 0x029BFCDB2DCE28D9, 0x55A06295CE870B07, 0x79BE667EF9DCBBAC];
const GY: [u64; 4] = [0x9C47D08FFB10D4B8, 0xFD17B448A6855419, 0x5DA4FBFC0E1108A8, 0x483ADA7726A3C465];
const BETA: [u64; 4] = [0xC1396C28719501EE, 0x9CF0497512F58995, 0x6E64479EAC3434E9, 0x7AE96A2B657C0710];
const LAMBDA: [u64; 4] = [0xDF02967C1B23BD72, 0x122E22EA20816678, 0xA5261C028812645A, 0x5363AD4CC05C30E0];
```

## WIF Format

```rust
// Mainnet compressed: version=0x80, suffix=0x01
// Testnet compressed: version=0xEF, suffix=0x01
fn to_wif(secret: &[u8; 32], mainnet: bool) -> String {
    let version = if mainnet { 0x80 } else { 0xEF };
    let mut data = vec![version];
    data.extend_from_slice(secret);
    data.push(0x01);
    let checksum = &sha256(sha256(&data))[..4];
    data.extend_from_slice(checksum);
    bs58::encode(data).into_string()
}
```

## Bech32 HRPs

```rust
const HRP_MAINNET: &str = "pocx";
const HRP_TESTNET: &str = "tpocx";
```

## Verifizierte Test-Vektoren

| Private Key | Address | WIF |
|------------|---------|-----|
| `2233181AC0DA99DC...` | `pocx1qmadf0xkl7qllnq06cprxehqeeenkn7hg26pgrf` | `KxNBzneS1F2AmRacLWfHa3XcoCPtKiZCtwEreW9kio4b9ewe5HzN` |
| `689FDD5BFAEB3F4D...` | `pocx1qmadf0xrcf4ljapsu23ppdcmammfet4jw555sk5` | `Kzj61Xuk6hra69TJWLPvapoKZMWb71vnQELbzCKEkAjh5zL78qFK` |
| `3932C5058E39F5A5...` | `pocx1qmadf0x5ulmn5vzv76c933hpgjjw5zwgvruqddw` | `Ky8tyZADEtBmx5wnArSETqnomGgx8Mg9pwCMSBcNB2oSmvs2urGS` |

## GPU Backend Selection Logic

```
1. If --backend=cuda ? CUDA only
2. If --backend=opencl ? OpenCL only  
3. Auto: Try CUDA ? Fallback OpenCL
```

## Kernel Output Structure

```rust
#[repr(C)]
struct GpuFoundItem {
    thread_id: u32,
    increment: i32,
    endomorphism: i32,  // 0, 1, 2
    hash160: [u8; 20],
}
```

## Private Key Recovery

```rust
k = base_key
k = if increment < 0 { n - (k + |increment|) } else { k + increment }
k = match endomorphism {
    1 => k * lambda % n,
    2 => k * lambda2 % n,
    _ => k
}
```

## Difficulty

```
pocx1qa     ? 2^5  = 32
pocx1qab    ? 2^10 = 1,024
pocx1qabc   ? 2^15 = 32,768
pocx1qabcd  ? 2^20 = 1,048,576
```

## Cargo Commands

```bash
# Build with both backends
cargo build --release

# Build CUDA only
cargo build --release --features cuda --no-default-features

# Build OpenCL only
cargo build --release --features opencl --no-default-features

# Run tests
cargo test

# Run CUDA/OpenCL parity tests
cargo test --features "cuda opencl" cuda_opencl_parity

# Benchmark
cargo bench
```
