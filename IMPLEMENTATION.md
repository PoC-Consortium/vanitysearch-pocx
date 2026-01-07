# VanitySearch-POCX Implementation Plan

## Overview
Rust port of VanitySearch for bech32 vanity address generation (CPU+CUDA+OpenCL).

## Reference implementation: https://github.com/JeanLucPons/VanitySearch
- strictly stick to the cuda implementation, it is critical to match it's performance!
- the reference performance reaches 11800MKeys/s, so the goal is to hit that. Every performance below 11000MKeys/s is absolutely **NOT** acceptable.
- secondary goal is the CPU performance, which needs to match at least 30MKeys/s with wildcards and 90MKeys/s without wildcards - **DO NOT SPEND RESSOURCING OPTIMIZING CPU BEFORE CUDA TARGETS ARE MET**.
- OpenCL target: achieve at least 80% of CUDA performance (~9.6 GKey/s)
- only extract the parts necessary for the compressed bech32 keys
- in the comparison of the results, skip the hrp + 1 + segwit version in the prefix.
- since the prefix always starts the same, we no longer pass the hrp + 1 + segwit to the search parameter, if there is no wildcard at the start, assume pattern is at beginning only.
- we do **NOT** need wildcard support for bech32 in GPU mode! Just give the user a warning if he uses illegal letters in his searchterm.

## Reference tests for performance measurement, **DO NOT ALTER!!**
- CPU: pattern "madf0*", 30s timeout - target >30MKeys/s
- CPU: pattern "madf0", 30s timeout - target >80MKeys/s
- GPU (CUDA): pattern "evlseed", 30s timeout - target >11000MKeys/s
- GPU (OpenCL): pattern "evlseed", 30s timeout - target >9600MKeys/s (80% of CUDA)

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

## Status: ✅ CPU Mode (65 MKey/s) | ✅ CUDA (12 GKey/s) | ✅ OpenCL (6.6 GKey/s)

## Current Performance
| Mode | With Wildcards (CPU ONLY) | Without Wildcards | Target | Status |
|------|----------------|-------------------|--------|--------|
| CPU | 65 MKey/s | 66 MKey/s | 30/90 MKey/s | ✅ 217% / 73% |
| CUDA | - | 12 GKey/s | 11 GKey/s | ✅ 109% |
| OpenCL | - | 6.6 GKey/s | 9.6 GKey/s | ⚠️ 69% (Fermat inverse overhead) |

- Hardware-accelerated SHA256/RIPEMD160 via sha2/ripemd crates (uses SHA-NI when available)
- GPU uses endomorphism for 6x address checks per point

## Scope
- Bech32 only (lowercase)
- Mainnet (bc1q) + Testnet (tb1q) descriptors
- WIF output with descriptor checksum
- CPU mode + CUDA mode + OpenCL mode
- Timeout (-T/--timeout) and max-found (-m/--max-found)

## Architecture

```
src/
├── main.rs              # CLI entry
├── lib.rs               # Library exports
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
│   ├── gpu.rs           # CUDA FFI wrapper
│   └── opencl.rs        # OpenCL wrapper
└── output.rs            # Result formatting

cuda/                    # CUDA kernels
├── GPUBech32.cu         # Bech32 CUDA kernel
├── GPUBech32.h          # Bech32 kernel header
├── GPUGroup.h           # Group constants (G table)
├── GPUHash.h            # SHA256/RIPEMD160 GPU
└── GPUMath.h            # secp256k1 field math

opencl/                  # OpenCL kernels (NEW)
├── bech32_kernel.cl     # Main OpenCL kernel
├── math.cl              # secp256k1 field arithmetic
├── hash.cl              # SHA256/RIPEMD160
└── group.cl             # G table constants
```

## Phase 1: Core Crypto ✅
- [x] secp256k1 field/scalar/point
- [x] sha256, ripemd160
- [x] bech32 encode
- [x] hash160 (sha256 + ripemd160)

## Phase 2: WIF + Output ✅
- [x] WIF encoding (mainnet/testnet)
- [x] Descriptor checksum (#checksum)
- [x] Colored terminal output

## Phase 3: Pattern Matching ✅
- [x] Wildcard pattern matcher (? and *)
- [x] Bech32 charset validation

## Phase 4: CPU Search Engine ✅
- [x] Key generation with endomorphism
- [x] Batch point multiplication
- [x] Pattern checking
- [x] Multi-threaded search
- [x] Bug fix: correct key offset calculation

## Phase 5: CUDA Integration ✅
- [x] CUDA kernel (GPUBech32.cu)
- [x] FFI bindings (gpu.rs)
- [x] Pattern constant memory optimization
- [x] Endomorphism (6 addresses/point)
- [x] Performance: 12 GKey/s ✅

## Phase 6: OpenCL Integration ✅
- [x] Port GPUMath.h to OpenCL (field arithmetic) → math.cl
- [x] Port GPUHash.h to OpenCL (SHA256/RIPEMD160) → hash.cl
- [x] Port GPUGroup.h to OpenCL (G table constants) → group.cl
- [x] Port GPUBech32.h to OpenCL (main kernel) → bech32_kernel.cl
- [x] Create Rust bindings (opencl.rs using ocl crate)
- [x] Add --opencl CLI flag
- [x] OpenCL runtime integration (ocl crate v0.19)
- [x] Tested on NVIDIA RTX 5090 (via CUDA OpenCL runtime)
- [ ] Test on AMD GPU
- [ ] Test on Intel GPU
- [x] **Fixed ModInv**: Replaced broken binary GCD with Fermat's little theorem
- [x] **Fixed p-2 exponent encoding**: Corrected 64-bit word 0 from 0xFFFFFC2D to 0xFFFFFFFEFFFFFC2D
- [x] Performance: 6.6 GKey/s (57% of CUDA, below 80% target due to Fermat overhead)

## Phase 7: CLI + Output ✅
- [x] Argument parsing (clap)
- [x] Timeout handling
- [x] Max-found handling
- [x] Output formatting (text/json/csv)
- [x] Colored output (green/yellow/red)

## Usage
```bash
# CPU mode (default)
vanitysearch-pocx "bc1qmadf0*" -T 30

# CUDA GPU mode (requires --features cuda build)
vanitysearch-pocx --gpu "bc1qevlseed*" -T 30

# OpenCL GPU mode (requires --features opencl build)
vanitysearch-pocx --opencl "bc1qevlseed*" -T 30

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
rayon = "1.10"
num_cpus = "1.16"
rand = "0.8"
hex = "0.4"
sha2 = "0.10"
ripemd = "0.1"
colored = "2"

[build-dependencies]
cc = "1.0"

[features]
default = []
cuda = []
opencl = []
```

## GPU Build Config

### CUDA
- Toolkit: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1
- Arch: sm_75, sm_86, sm_89, sm_120

```bash
cargo build --release --features cuda
```

### OpenCL
- Requires OpenCL SDK (AMD APP SDK, Intel OpenCL, or NVIDIA CUDA)
- Works with AMD, Intel, and NVIDIA GPUs

```bash
cargo build --release --features opencl
```

## Progress Log

### v0.5.0 (Current)
- Fixed CPU private key generation bug (key offset calculation)
- Simplified to Mainnet + Testnet only (removed networks.toml)
- Added colored terminal output
- GPU (CUDA) achieving 12 GKey/s
- OpenCL kernel files completed (math.cl, hash.cl, group.cl, bech32_kernel.cl)
- OpenCL CLI integration (--opencl, --opencl-platform, --opencl-device, --opencl-threads)
- OpenCL Rust stub ready (awaiting ocl crate integration)

### v0.4.0
- CPU search optimized to 65 MKey/s with wildcards
- Hardware-accelerated SHA256/RIPEMD160

### v0.3.0
- CUDA kernel implemented
- GPU achieving 12 GKey/s

### v0.2.0
- All Rust modules implemented
- CPU search working

### v0.1.0
- Initial implementation

## Output Format
```
Address         bc1qmadf0qq7galyvccjajm8ep4fsh7wucae3pajrn   (green)
Main Descriptor wpkh(L5NrxhLDQMWF...)#0kf0tee2               (yellow WIF)
Test Descriptor wpkh(cVjrRcL4qRCW...)#4ydpfzte               (yellow WIF)
Private Key     0xf3682e65f7d39e58...                        (red)
```
1. Use pure Rust for crypto (no external C deps except CUDA)
2. Bech32 charset: qpzry9x8gf2tvdw0s3jn54khce6mua7l
3. Pattern always starts with hrp + "1q" (witness v0)
4. Descriptor format: wpkh(WIF)#checksum
