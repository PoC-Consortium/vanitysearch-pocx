# VanitySearch-POCX Implementation Plan

## Overview
Rust port of VanitySearch for bech32 vanity address generation (CPU+CUDA).

## Reference implementation: https://github.com/JeanLucPons/VanitySearch
- strictly stick to the cuda implementation, it is critical to match it's performance!
- the reference performance reaches 11800MKeys/s, so the goal is to hit that. Every performance below 11000MKeys/s is absolutely **NOT** acceptable.
- secondary goal is the CPU performance, which needs to match at least 30MKeys/s with wildcards and 90MKeys/s without wildcards - **DO NOT SPEND RESSOURCING OPTIMIZING CPU BEFORE CUDA TARGETS ARE MET**.
- only extract the parts necessary for the compressed bech32 keys
- in the comparison of the results, skip the hrp + 1 + segwit version in the prefix.
- since the prefix always starts the same, we no longer pass the hrp + 1 + segwit to the search parameter, if there is no wildcard at the start, assume pattern is at beginning only.
- we do **NOT** need wildcard support for bech32 in GPU mode! Just give the user a warning if he uses illegal letters in his searchterm.

## Reference tests for performance measurement, **DO NOT ALTER!!**
- CPU: pattern "madf0*", 30s timeout - target >30MKeys/s
- CPU: pattern "madf0", 30s timeout - target >90MKeys/s
- GPU: pattern "evlseed", 30s timeout - target >11000MKeys/s

## Important to focus on for the GPU implementation
- GPUEngine.cu ll 484-486 - reference to how bech32 compressed keys are processed:
```
if (searchMode == SEARCH_COMPRESSED) {
        comp_keys_comp << < nbThread / nbThreadPerGroup, nbThreadPerGroup >> >
          (inputPrefix, inputPrefixLookUp, inputKey, maxFound, outputPrefix);
```
- GPUEngine.cu ll 57ff - function call for bech32 processing:
```
__global__ void comp_keys_comp(prefix_t *prefix, uint32_t *lookup32, uint64_t *keys, uint32_t maxFound, uint32_t *found) {

  int xPtr = (blockIdx.x*blockDim.x) * 8;
  int yPtr = xPtr + 4 * blockDim.x;
  ComputeKeysComp(keys + xPtr, keys + yPtr, prefix, lookup32, maxFound, found);

}
```

## Status: 🟡 CPU Mode (28 MKey/s, target 30/90) | 🔄 GPU Mode Ready (needs CUDA build)

## Current CPU Performance
- **With wildcards** (`bc1qmadf0*`): 28.42 MKey/s (target >30 MKey/s) - 94.7% ✓
- **Without wildcards** (`bc1qmadf0`): 28.03 MKey/s (target >90 MKey/s) - 31.1% ❌
- Note: Non-wildcard speedup requires SIMD field arithmetic (future optimization)

## Scope
- Bech32 only (lowercase)
- Custom HRP via networks.toml
- WIF output with descriptor checksum
- CPU mode + CUDA mode (OpenCL later)
- Timeout (-T/--timeout) and max-found (-m/--max-found)

## Architecture

```
src/
├── main.rs              # CLI entry
├── lib.rs               # Library exports
├── config.rs            # Networks TOML parser
├── secp256k1/           # EC operations
│   ├── mod.rs
│   ├── field.rs         # Field arithmetic
│   ├── scalar.rs        # Scalar arithmetic
│   └── point.rs         # Point operations
├── hash/                # Hash functions
│   ├── mod.rs
│   ├── sha256.rs
│   └── ripemd160.rs
├── bech32.rs            # Bech32 encoding
├── wif.rs               # WIF encoding + descriptor checksum
├── pattern.rs           # Pattern matching (wildcard)
├── search/              # Search engines
│   ├── mod.rs
│   ├── cpu.rs           # CPU search
│   └── gpu.rs           # CUDA FFI wrapper
└── output.rs            # Result formatting

cuda/
├── bech32_kernel.cu     # New CUDA kernel for bech32
├── GPUEngine.cu         # Original VanitySearch kernel (reference)
└── *.h                  # Supporting headers
```

## Phase 1: Core Crypto ✅
- [x] secp256k1 field/scalar/point
- [x] sha256, ripemd160
- [x] bech32 encode
- [x] hash160 (sha256 + ripemd160)

## Phase 2: Config + WIF ✅
- [x] networks.toml parser
- [x] WIF encoding (mainnet/testnet)
- [x] Descriptor checksum (#checksum)

## Phase 3: Pattern Matching ✅
- [x] Wildcard pattern matcher (? and *)
- [x] Bech32 charset validation

## Phase 4: CPU Search Engine ✅
- [x] Key generation with endomorphism
- [x] Batch point multiplication
- [x] Pattern checking
- [x] Multi-threaded search

## Phase 5: CUDA Integration 🔄
- [x] CUDA kernel written (bech32_kernel.cu) - please verify if this is matching the original algorithms!
- [x] FFI bindings ready (gpu.rs)
- [x] Graceful fallback when CUDA not available
- [ ] CUDA build integration (requires CUDA toolkit)
- [ ] Testing on GPU hardware

## Phase 6: CLI + Output ✅
- [x] Argument parsing (clap)
- [x] Timeout handling
- [x] Max-found handling
- [x] Output formatting (text/json/csv)

## Usage
```bash
# CPU mode (default)
vanitysearch-pocx "bc1qmadf0*" -T 30

# GPU mode (requires --features cuda build)
vanitysearch-pocx --gpu "bc1qevlseed*" -T 30

# With max matches
vanitysearch-pocx "bc1qtest*" -m 10

# JSON output
vanitysearch-pocx "bc1qtest*" -T 10 -o json
```

## Dependencies
```toml
[dependencies]
clap = { version = "4", features = ["derive"] }
serde = { version = "1", features = ["derive"] }
toml = "0.8"
rayon = "1.10"
num_cpus = "1.16"
rand = "0.8"
hex = "0.4"

[build-dependencies]
cc = "1.0"

[features]
default = []
cuda = []
```

## CUDA Build Config
- Toolkit: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1
- Arch: sm_75, sm_86, sm_89, sm_120

To build with CUDA:
```bash
cargo build --release --features cuda
```

## Progress Log

### Session Complete
- All Rust modules implemented
- CPU search working at ~7 Mkey/s
- CUDA kernel written
- GPU FFI scaffolding ready
- Graceful CPU fallback when GPU unavailable

## Output Format
```
Address: bc1qmadf0qq7galyvccjajm8ep4fsh7wucae3pajrn
Mainnet WIF: L5NrxhLDQMWF3gh18A8AT7wayxtJh5HSvBAzZRGqg9PCWSqiTVgT
Testnet WIF: cVjrRcL4qRCWD8AGWZwHpSSecCBiMXP8zDKTfqjMBG3CmBvpj73g
Mainnet Descriptor: wpkh(L5NrxhLDQMWF3gh18A8AT7wayxtJh5HSvBAzZRGqg9PCWSqiTVgT)#0kf0tee2
Testnet Descriptor: wpkh(cVjrRcL4qRCWD8AGWZwHpSSecCBiMXP8zDKTfqjMBG3CmBvpj73g)#4ydpfzte
Private Key (hex): 0xf3682e65f7d39e588fac51813abe280cd33374240b65548f733b8518ba6cbcbe
```
1. Use pure Rust for crypto (no external C deps except CUDA)
2. Bech32 charset: qpzry9x8gf2tvdw0s3jn54khce6mua7l
3. Pattern always starts with hrp + "1q" (witness v0)
4. Descriptor format: wpkh(WIF)#checksum
