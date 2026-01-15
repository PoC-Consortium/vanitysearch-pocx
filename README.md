# VanitySearch-POCX

High-performance bech32 vanity address generator with GPU acceleration. Generate custom addresses with specific prefixes for **any blockchain using bech32 encoding** (Bitcoin, Litecoin, POCX, etc).

[![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](LICENSE)

## Features

- 🚀 **Multi-platform**: CPU, CUDA (NVIDIA), and OpenCL (AMD/Intel/NVIDIA) support
- 🌐 **Universal bech32**: Works with any bech32 network (bc1, ltc1, pocx1, tpocx1, etc)
- ⚡ **High Performance**: Up to 12 GKey/s on CUDA, 66 MKey/s on CPU
- 🎯 **Wildcard Patterns**: Use `*` and `?` for flexible pattern matching (CPU mode only)
- 🔐 **Complete Output**: Generates private key, WIF, and descriptor checksum
- 🎨 **Colored Output**: Easy-to-read terminal output with syntax highlighting
- 🐳 **Docker Ready**: Pre-built images with CUDA/OpenCL support

## Supported Networks

This tool works with **any bech32-encoded cryptocurrency**:

- **POCX**: `pocx1q...` (mainnet), `tpocx1q...` (testnet)
- **Bitcoin**: `bc1q...` (mainnet), `tb1q...` (testnet)
- **Litecoin**: `ltc1q...` (mainnet), `tltc1q...` (testnet)
- **Any other bech32 coin**: Just specify the HRP (human-readable part)

The pattern you search for will include the HRP automatically based on network detection.

## Performance

| Mode | Speed | Platform |
|------|-------|----------|
| CUDA | 12.5 GKey/s | NVIDIA RTX 5090 |
| OpenCL | 8.6 GKey/s | NVIDIA RTX 5090 |
| CPU (no wildcards) | 66 MKey/s | Multi-core CPU |
| CPU (wildcards) | 65 MKey/s | Multi-core CPU |

## Installation

### Quick Start with Docker (Recommended)

The easiest way to get started with GPU support:

```bash
# Pull the Docker image with CUDA/OpenCL support
docker pull ghcr.io/PoC-Consortium/vanitysearch-pocx:latest

# Run with NVIDIA GPU
docker run --gpus all ghcr.io/PoC-Consortium/vanitysearch-pocx:latest "evlseed" -g -v -m 1

# Run with AMD/Intel GPU (OpenCL)
docker run --device=/dev/dri ghcr.io/PoC-Consortium/vanitysearch-pocx:latest "evlseed" -g --opencl -v -m 1

# CPU mode
docker run ghcr.io/PoC-Consortium/vanitysearch-pocx:latest "ev?seed*" -v -m 1
```

See [Docker Usage](#docker-usage) section below for more details.

### Prerequisites for Building

- Rust 1.70+ ([Install Rust](https://rustup.rs/))
- **For CUDA**: NVIDIA GPU + CUDA Toolkit 11.0+ ([Download](https://developer.nvidia.com/cuda-downloads))
- **For OpenCL**: OpenCL SDK (comes with GPU drivers)

### Build from Source

```bash
# Clone the repository
git clone https://github.com/PoC-Consortium/vanitysearch-pocx.git
cd vanitysearch-pocx

# Build with all features (CUDA + OpenCL)
cargo build --release

# CPU-only build
cargo build --release --no-default-features

# CUDA only
cargo build --release --no-default-features --features cuda

# OpenCL only
cargo build --release --no-default-features --features opencl
```

The compiled binary will be at `target/release/vanitysearch-pocx` (or `.exe` on Windows).

## Usage

### Basic Examples

```bash
# CPU mode - find 1 address starting with "pocx1qevlseed"
vanitysearch-pocx "evlseed" -v -m 1

# GPU mode - auto-detects CUDA, falls back to OpenCL
vanitysearch-pocx "evlseed" -g -v -m 1

# Force OpenCL mode (useful for NVIDIA cards to test OpenCL)
vanitysearch-pocx "evlseed" --opencl -v -m 1

# With wildcards (CPU only) - "evlseed" anywhere in the address (could be at end in checksum, provide HRP prefix in this case)
vanitysearch-pocx "pocx1q*evlseed*" -v -m 1

# Wildcards - any 3 chars then "test"
vanitysearch-pocx "???test*" -v -m 1

# With 60 second timeout, output every found address asap (-v)
vanitysearch-pocx "evlseed" -v -T 60

# Find multiple matches
vanitysearch-pocx "evlseed" -v -m 5

# Default mode - shows addresses during search, full details at end
vanitysearch-pocx "evlseed" -T 10
```

### Command-Line Options

```
vanitysearch-pocx [OPTIONS] <PATTERN>

Arguments:
  <PATTERN>  Search pattern (e.g., "test", "madf0*", "???abc")

Options:
  -g, --gpu               Use GPU mode (auto-detects CUDA, falls back to OpenCL)
      --opencl            Force OpenCL mode
      --gpu-id <N>        GPU device ID (default: 0)
      --gpu-threads <N>   Number of GPU threads (default: auto-detect)
      --gpu-block-size <N> Threads per block for CUDA (default: 256)
      --opencl-platform <N> OpenCL platform index (default: 0)
      --opencl-threads <N>  OpenCL work items (default: 65536)
  -t, --threads <N>       CPU thread count (default: auto-detect)
  -T, --timeout <SEC>     Stop after N seconds
  -m, --max-found <N>     Stop after finding N matches
  -o, --output <FMT>      Output format: text, json, csv (default: text)
  -f, --output-file <FILE> Write output to file
  -q, --quiet             Suppress progress output
  -v, --verbose           Print full details for each match immediately
  -h, --help              Print help
  -V, --version           Print version
```

### Output Modes

**Default Mode**: Shows address when found, full details at the end
```bash
vanitysearch-pocx "test" -T 10
# Output during search:
#   Found: pocx1qtest...
# Output after search:
#   Full details (address, descriptors, private key)
```

**Verbose Mode** (`-v`): Shows full details immediately when each match is found
```bash
vanitysearch-pocx "test" -v -T 10
# Output when each match is found:
# Address         pocx1qev7seeducvqaqwkc8txav6hxl7ga2dxntg5w3c
# Main Descriptor wpkh(KwFK92Lbitdf9GLVLdijEgsTtYVrjfbxpp1QmdMkQzCx4HFyr9Pk)#eafg539f  
# Test Descriptor wpkh(cMcJbwLT9xKvJhokj3Xrc1NXWmoGQ7hetr9st3pFv6rxK2HwT6rK)#wpwlyx4e  
# Private Key     0x00d310937c09fa6f154ec1e93bdd7d5672be2c16e11277c0e075989b69e75ae0
```

**Quiet Mode** (`-q`): Suppresses all progress output, only shows final results

### GPU Mode Behavior

The `-g` flag automatically detects available GPU backends:
1. **CUDA** is tried first (NVIDIA GPUs)
2. **OpenCL** is used as fallback (AMD/Intel/NVIDIA GPUs)
3. Use `--opencl` to force OpenCL even when CUDA is available

**Note**: GPU modes do not support wildcard patterns (`*` or `?`). Use CPU mode for wildcards.

### Pattern Format

Patterns are matched against the address **after** the HRP and separator:
- `pocx1q` on mainnet → search pattern starts after `pocx1q`
- `tpocx1q` on testnet → search pattern starts after `tpocx1q`

**Valid characters**: `qpzry9x8gf2tvdw0s3jn54khce6mua7l` (bech32 charset, lowercase only)

**Wildcards** (CPU mode only):
- `*` = any number of characters
- `?` = exactly one character

**Examples**:
- `test` → finds `pocx1qtest...`, `tpocx1qtest...`
- `test*` → finds addresses starting with `pocx1qtest...`, `tpocx1qtest...`
- `???abc` → finds addresses like `pocx1qXYZabc...` (X, Y, Z = any lowercase chars)

## Output Format

```
Address         pocx1qcudarun4409ey62fydetrpwhg7lk593fhnrwk0
Main Descriptor wpkh(KyTmdk1tQT7G2zy9Ccro1wFsqqyrYsZoWb6ucyADMLGuzQqxZ3EY)#r750rech
Test Descriptor wpkh(cPpm6f1jqWoXCSSQb2fvPFkwU5HGDKfVadFNjPcirSvvF9xqGYF3)#wzxmwkek
Private Key     0x42e8534f1358f839bd0d3587561f2ca7564ad44163e9558e7490b440743c9b88
```

- **Address**:         The generated vanity address
- **Main Descriptor**: Mainnet WIF with descriptor checksum
- **Test Descriptor**: Testnet WIF with descriptor checksum  
- **Private Key**:     Raw 256-bit private key in hexadecimal

## Docker Usage

### Quick Start

```bash
# Pull the latest image
docker pull ghcr.io/PoC-Consortium/vanitysearch-pocx:latest

# Run with NVIDIA GPU (CUDA)
docker run --gpus all ghcr.io/PoC-Consortium/vanitysearch-pocx:latest "test" -g -v -m 1

# Run with AMD/Intel GPU (OpenCL)
docker run --device=/dev/dri ghcr.io/PoC-Consortium/vanitysearch-pocx:latest "test" -g -v -m 1

# CPU mode
docker run ghcr.io/PoC-Consortium/vanitysearch-pocx:latest "test*" -v -m 1
```

### Windows with NVIDIA

```powershell
# Install Docker Desktop with WSL2
# Install NVIDIA Container Toolkit

docker run --gpus all ghcr.io/PoC-Consortium/vanitysearch-pocx:latest "evlseed" -g -v -m 1
```

### Linux with NVIDIA

```bash
# Install nvidia-container-toolkit
# See: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

docker run --gpus all ghcr.io/PoC-Consortium/vanitysearch-pocx:latest "madf0" -g -v -m 1
```

### Linux with AMD GPU

```bash
# AMD GPU access via /dev/dri
docker run --device=/dev/dri --group-add video \
  ghcr.io/PoC-Consortium/vanitysearch-pocx:latest "test" -g -v -m 1
```

### Save Output to File

```bash
# Redirect output to host file
docker run --gpus all ghcr.io/PoC-Consortium/vanitysearch-pocx:latest \
  "evlseed" -g -T 60 > output.txt
```

### Build Docker Image Locally

```bash
# Build multi-arch image with CUDA/OpenCL support
docker build -t vanitysearch-pocx:local .

# Run local build
docker run --gpus all vanitysearch-pocx:local "test" -g -v -m 1
```

## GPU Mode Details

### CUDA vs OpenCL

**CUDA** (NVIDIA only):
- ✅ Fastest performance (~12 GKey/s on RTX 5090)
- ✅ Native NVIDIA optimization
- ❌ NVIDIA GPUs only

**OpenCL** (Cross-platform):
- ✅ Works on AMD, Intel, and NVIDIA GPUs
- ✅ Single codebase for all vendors
- ⚠️ ~55% of CUDA performance on NVIDIA (due to Fermat inverse overhead)
- 🔮 May perform better on native AMD/Intel OpenCL implementations

### GPU Limitations

**Wildcards not supported in GPU mode**. GPU mode performs **exact prefix matching** for maximum performance. If you need wildcards, use CPU mode.

Example:
```bash
# ✅ GPU mode - exact prefix
vanitysearch-pocx "evlseed" -g -v -m 1

# ❌ GPU mode - wildcards will show warning
vanitysearch-pocx "evl*" -g  # Use CPU mode instead

# ✅ CPU mode - wildcards work
vanitysearch-pocx "evl*" -v -m 1
```

### Selecting OpenCL Device

```bash
# List platforms and devices (shows in startup output)
vanitysearch-pocx "test" --opencl -v -m 1

# Use specific platform and device
vanitysearch-pocx "test" --opencl --opencl-platform 0 --opencl-device 1 -v -m 1
```

## Difficulty & Search Time

Address difficulty grows exponentially with pattern length:

| Pattern Length | Difficulty | Est. Time (12 GKey/s GPU) |
|----------------|------------|---------------------------|
| 5 chars | 2^25 (~33M) | < 1 second |
| 7 chars | 2^35 (~34B) | ~3 seconds |
| 9 chars | 2^45 (~35T) | ~50 minutes |
| 11 chars | 2^55 (~36P) | ~40 days |

*Note: These are statistical averages. Actual time varies.*

## Security Notice

⚠️ **IMPORTANT**: Keep your private keys secure!

- Never share your private key with anyone
- Store keys in a secure password manager or hardware wallet
- Test with small amounts first on testnet
- Back up your keys before sending funds to generated addresses

## Technical Details

### Algorithm

- **Elliptic Curve**: secp256k1
- **Address Format**: Bech32 (witness v0)
- **Optimization**: Endomorphism for 6 addresses per point evaluation
- **GPU Strategy**: Batch point multiplication with Montgomery's trick for modular inverse

### Architecture

- **Core Crypto**: Pure Rust implementation of secp256k1
- **Hashing**: Hardware-accelerated SHA-256 and RIPEMD-160 (via `sha2`/`ripemd` crates)
- **CUDA Kernel**: Direct port of VanitySearch's proven GPU algorithm
- **OpenCL Kernel**: Cross-platform port with Fermat's little theorem for modular inverse

## Benchmarking

Test performance on your hardware:

```bash
# CPU benchmark (30 seconds)
vanitysearch-pocx "cpurun" -T 30

# CUDA benchmark (30 seconds)
vanitysearch-pocx "cudarun" -T 30 -g

# OpenCL benchmark (30 seconds)
vanitysearch-pocx "0pencl3" -T 30 -g --opencl
```

## Troubleshooting

### CUDA Issues

**Error: "CUDA not found"**
- Install CUDA Toolkit from NVIDIA
- Set `CUDA_PATH` environment variable
- Rebuild with `--features cuda`

**Low Performance**
- Close other GPU-intensive applications
- Update NVIDIA drivers
- Check GPU utilization with `nvidia-smi`

### OpenCL Issues

**Error: "No OpenCL platforms found"**
- Install GPU vendor's OpenCL SDK or drivers
- AMD: Install AMD APP SDK
- Intel: Install Intel OpenCL runtime
- NVIDIA: CUDA toolkit includes OpenCL

**Wrong Device Selected**
- Use `--opencl-platform` and `--opencl-device` to select correct GPU
- Check startup output for available devices

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

### Development Setup

```bash
# Clone repository
git clone https://github.com/PoC-Consortium/vanitysearch-pocx.git
cd vanitysearch-pocx

# Build with all features
cargo build --release

# Run tests
cargo test

# Run with logging
RUST_LOG=debug cargo run --release -- "test" -T 10
```

### Creating a Release

Releases are automatically built and published when you push a version tag:

```bash
# Update version in Cargo.toml
# Commit changes
git commit -am "Bump version to 0.5.1"

# Create and push tag
git tag v0.5.1
git push origin main --tags
```

This triggers GitHub Actions to:
- Build binaries for Linux, Windows, and macOS
- Build and push Docker images to GitHub Container Registry
- Create a GitHub release with all artifacts

## License

This project is licensed under the GPL-3.0 License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Original [VanitySearch](https://github.com/JeanLucPons/VanitySearch) by Jean Luc PONS
- Bitcoin Core for secp256k1 implementation reference
- POC blockchain community

## Disclaimer

This tool is for generating vanity addresses. Always verify generated addresses and test with small amounts first. The authors are not responsible for any loss of funds due to misuse or bugs.

---

**Star ⭐ this repo if you find it useful!**
