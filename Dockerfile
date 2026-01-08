# Multi-stage Dockerfile for VanitySearch-POCX with CUDA and OpenCL support
# Supports NVIDIA (CUDA/OpenCL), AMD (OpenCL), and Intel (OpenCL) GPUs

# Stage 1: Build stage with CUDA toolkit
FROM nvidia/cuda:13.1.0-devel-ubuntu22.04 AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    pkg-config \
    libssl-dev \
    ocl-icd-opencl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Set working directory
WORKDIR /build

# Copy source code
COPY Cargo.toml Cargo.lock build.rs ./
COPY src ./src
COPY cuda ./cuda
COPY opencl ./opencl

# Build release binary with both CUDA and OpenCL support
RUN cargo build --release --features cuda,opencl

# Stage 2: Runtime stage with minimal CUDA runtime and OpenCL
FROM nvidia/cuda:13.1.0-runtime-ubuntu22.04

# Install OpenCL ICD loader and drivers
RUN apt-get update && apt-get install -y \
    ocl-icd-libopencl1 \
    ocl-icd-opencl-dev \
    clinfo \
    && rm -rf /var/lib/apt/lists/*

# Create OpenCL vendor directory and register NVIDIA
RUN mkdir -p /etc/OpenCL/vendors && \
    echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

# Copy the built binary from builder stage
COPY --from=builder /build/target/release/vanitysearch-pocx /usr/local/bin/vanitysearch-pocx

# Set up working directory
WORKDIR /work

# Set entrypoint to the binary
ENTRYPOINT ["/usr/local/bin/vanitysearch-pocx"]

# Default command shows help
CMD ["--help"]

# Labels
LABEL org.opencontainers.image.title="VanitySearch-POCX"
LABEL org.opencontainers.image.description="High-performance bech32 vanity address generator with GPU support"
LABEL org.opencontainers.image.licenses="GPL-3.0"
LABEL org.opencontainers.image.source="https://github.com/yourusername/vanitysearch-pocx"

# Usage examples:
# docker run --gpus all vanitysearch-pocx "test" -g
# docker run --device=/dev/dri vanitysearch-pocx "test" -g --opencl
# docker run vanitysearch-pocx "test*" -T 30
