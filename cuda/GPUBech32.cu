/*
 * VanitySearch-POCX Bech32 CUDA Kernel
 * Based on VanitySearch by Jean Luc PONS
 * 
 * This kernel searches for bech32 vanity addresses (bc1q...) 
 * using compressed public keys only.
 */

#ifdef _WIN32
#include <stdio.h>
#include <string.h>
#else
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#endif

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

// Include VanitySearch GPU components
#include "GPUGroup.h"
#include "GPUMath.h"
#include "GPUHash.h"
#include "GPUBech32.h"

// Constant memory for pattern (fast, cached)
__constant__ uint8_t d_pattern_const[32];
__constant__ int d_pattern_len;

// ---------------------------------------------------------------------------------------
// Kernel entry point for bech32 search
// ---------------------------------------------------------------------------------------

__global__ void bech32_search_kernel(
    uint64_t *keys,           // Input: starting points (x, y interleaved)
    uint32_t maxFound,        // Max matches to record
    uint32_t *found           // Output: match count + match data
) {
    int xPtr = (blockIdx.x * blockDim.x) * 8;
    int yPtr = xPtr + 4 * blockDim.x;
    ComputeKeysBech32(
        keys + xPtr, 
        keys + yPtr, 
        d_pattern_const,  // Use constant memory
        d_pattern_len,    // Use constant memory
        maxFound, 
        found
    );
}

// ---------------------------------------------------------------------------------------
// Host-side helper functions (extern "C" for Rust FFI)
// ---------------------------------------------------------------------------------------

extern "C" {

// Get CUDA device count
int cuda_bech32_get_device_count() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        return 0;
    }
    return count;
}

// Get device name
void cuda_bech32_get_device_name(int deviceId, char *name, int maxLen) {
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, deviceId) == cudaSuccess) {
        strncpy(name, prop.name, maxLen - 1);
        name[maxLen - 1] = '\0';
    } else {
        name[0] = '\0';
    }
}

// Get device memory info
int cuda_bech32_get_device_memory(int deviceId, size_t *totalMem, size_t *freeMem) {
    cudaError_t err = cudaSetDevice(deviceId);
    if (err != cudaSuccess) return -1;
    
    err = cudaMemGetInfo(freeMem, totalMem);
    return (err == cudaSuccess) ? 0 : -1;
}

// Get device SM count (multiprocessor count)
int cuda_bech32_get_device_sm_count(int deviceId) {
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, deviceId) == cudaSuccess) {
        return prop.multiProcessorCount;
    }
    return 0;
}

// Initialize CUDA device
int cuda_bech32_init(int deviceId) {
    cudaError_t err = cudaSetDevice(deviceId);
    if (err != cudaSuccess) {
        printf("CUDA Error: Failed to set device %d: %s\n", deviceId, cudaGetErrorString(err));
        return -1;
    }
    
    // Get device properties
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, deviceId);
    if (err != cudaSuccess) {
        printf("CUDA Error: Failed to get device properties: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    printf("GPU: %s\n", prop.name);
    printf("  SM Count: %d\n", prop.multiProcessorCount);
    printf("  Memory: %.1f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
    
    return 0;
}

// GPU context structure
typedef struct {
    uint64_t *d_keys;         // Device: starting points
    uint8_t *d_pattern;       // Device: pattern (5-bit values)
    uint32_t *d_output;       // Device: output buffer
    int patternLen;           // Pattern length
    int numThreadGroups;      // Number of thread groups
    int threadsPerGroup;      // Threads per group
    size_t keysSize;          // Size of keys buffer
    size_t outputSize;        // Size of output buffer
} GpuContext;

// Create GPU context
GpuContext* cuda_bech32_create_context(
    int numThreadGroups,
    int threadsPerGroup,
    int maxFound
) {
    GpuContext *ctx = (GpuContext*)malloc(sizeof(GpuContext));
    if (!ctx) return NULL;
    
    ctx->numThreadGroups = numThreadGroups;
    ctx->threadsPerGroup = threadsPerGroup;
    ctx->patternLen = 0;
    
    // Each thread needs 8 uint64_t for (x, y) coordinates
    // Total threads = numThreadGroups * threadsPerGroup
    int totalThreads = numThreadGroups * threadsPerGroup;
    ctx->keysSize = totalThreads * 8 * sizeof(uint64_t);
    
    // Output buffer: count + maxFound * ITEM_SIZE32_BECH32 uint32
    ctx->outputSize = (1 + maxFound * ITEM_SIZE32_BECH32) * sizeof(uint32_t);
    
    cudaError_t err;
    
    // Allocate device memory
    err = cudaMalloc(&ctx->d_keys, ctx->keysSize);
    if (err != cudaSuccess) {
        printf("CUDA Error: Failed to allocate keys: %s\n", cudaGetErrorString(err));
        free(ctx);
        return NULL;
    }
    
    err = cudaMalloc(&ctx->d_pattern, 32);  // Max 32 x 5-bit values
    if (err != cudaSuccess) {
        printf("CUDA Error: Failed to allocate pattern: %s\n", cudaGetErrorString(err));
        cudaFree(ctx->d_keys);
        free(ctx);
        return NULL;
    }
    
    err = cudaMalloc(&ctx->d_output, ctx->outputSize);
    if (err != cudaSuccess) {
        printf("CUDA Error: Failed to allocate output: %s\n", cudaGetErrorString(err));
        cudaFree(ctx->d_keys);
        cudaFree(ctx->d_pattern);
        free(ctx);
        return NULL;
    }
    
    return ctx;
}

// Destroy GPU context
void cuda_bech32_destroy_context(GpuContext *ctx) {
    if (ctx) {
        if (ctx->d_keys) cudaFree(ctx->d_keys);
        if (ctx->d_pattern) cudaFree(ctx->d_pattern);
        if (ctx->d_output) cudaFree(ctx->d_output);
        free(ctx);
    }
}

// Set pattern (convert string to 5-bit values)
int cuda_bech32_set_pattern(GpuContext *ctx, const char *pattern, int len) {
    if (!ctx || !pattern || len < 0 || len > 32) return -1;
    
    uint8_t pattern5bit[32] = {0};
    
    // Convert pattern chars to 5-bit values
    for (int i = 0; i < len; i++) {
        char c = pattern[i];
        if (c >= 0 && c < 128) {
            int8_t val = -1;
            // Manual lookup since we're on host
            switch (c) {
                case 'q': case 'Q': val = 0; break;
                case 'p': case 'P': val = 1; break;
                case 'z': case 'Z': val = 2; break;
                case 'r': case 'R': val = 3; break;
                case 'y': case 'Y': val = 4; break;
                case '9': val = 5; break;
                case 'x': case 'X': val = 6; break;
                case '8': val = 7; break;
                case 'g': case 'G': val = 8; break;
                case 'f': case 'F': val = 9; break;
                case '2': val = 10; break;
                case 't': case 'T': val = 11; break;
                case 'v': case 'V': val = 12; break;
                case 'd': case 'D': val = 13; break;
                case 'w': case 'W': val = 14; break;
                case '0': val = 15; break;
                case 's': case 'S': val = 16; break;
                case '3': val = 17; break;
                case 'j': case 'J': val = 18; break;
                case 'n': case 'N': val = 19; break;
                case '5': val = 20; break;
                case '4': val = 21; break;
                case 'k': case 'K': val = 22; break;
                case 'h': case 'H': val = 23; break;
                case 'c': case 'C': val = 24; break;
                case 'e': case 'E': val = 25; break;
                case '6': val = 26; break;
                case 'm': case 'M': val = 27; break;
                case 'u': case 'U': val = 28; break;
                case 'a': case 'A': val = 29; break;
                case '7': val = 30; break;
                case 'l': case 'L': val = 31; break;
                default: return -2;  // Invalid character
            }
            pattern5bit[i] = (uint8_t)val;
        } else {
            return -2;  // Invalid character
        }
    }
    
    ctx->patternLen = len;
    
    // Copy pattern to constant memory (fast, cached)
    cudaError_t err = cudaMemcpyToSymbol(d_pattern_const, pattern5bit, 32);
    if (err != cudaSuccess) {
        printf("CUDA Error: Failed to copy pattern to constant memory: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    // Copy pattern length to constant memory
    err = cudaMemcpyToSymbol(d_pattern_len, &len, sizeof(int));
    if (err != cudaSuccess) {
        printf("CUDA Error: Failed to copy pattern length to constant memory: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    // Also keep in global memory for backward compatibility
    err = cudaMemcpy(ctx->d_pattern, pattern5bit, 32, cudaMemcpyHostToDevice);
    return (err == cudaSuccess) ? 0 : -1;
}

// Copy starting keys to device
int cuda_bech32_copy_keys(GpuContext *ctx, const uint8_t *keys, size_t size) {
    if (!ctx || !keys || size > ctx->keysSize) return -1;
    
    cudaError_t err = cudaMemcpy(ctx->d_keys, keys, size, cudaMemcpyHostToDevice);
    return (err == cudaSuccess) ? 0 : -1;
}

// Clear output buffer
int cuda_bech32_clear_output(GpuContext *ctx) {
    if (!ctx) return -1;
    
    cudaError_t err = cudaMemset(ctx->d_output, 0, ctx->outputSize);
    return (err == cudaSuccess) ? 0 : -1;
}

// Launch kernel
int cuda_bech32_launch(GpuContext *ctx, int maxFound) {
    if (!ctx) return -1;
    
    // Clear output count
    uint32_t zero = 0;
    cudaMemcpy(ctx->d_output, &zero, sizeof(uint32_t), cudaMemcpyHostToDevice);
    
    // Launch kernel
    dim3 grid(ctx->numThreadGroups);
    dim3 block(ctx->threadsPerGroup);
    
    bech32_search_kernel<<<grid, block>>>(
        ctx->d_keys,
        maxFound,
        ctx->d_output
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Kernel Error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    return 0;
}

// Synchronize and check for errors
int cuda_bech32_sync() {
    cudaError_t err = cudaDeviceSynchronize();
    return (err == cudaSuccess) ? 0 : -1;
}

// Get results
int cuda_bech32_get_results(
    GpuContext *ctx,
    uint32_t *matchCount,
    uint32_t *results,
    int maxResults
) {
    if (!ctx || !matchCount) return -1;
    
    // Get match count
    cudaError_t err = cudaMemcpy(matchCount, ctx->d_output, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) return -1;
    
    // Get match data
    if (*matchCount > 0 && results) {
        int count = (*matchCount < (uint32_t)maxResults) ? *matchCount : maxResults;
        size_t dataSize = count * ITEM_SIZE32_BECH32 * sizeof(uint32_t);
        err = cudaMemcpy(results, ctx->d_output + 1, dataSize, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) return -1;
    }
    
    return 0;
}

// Get updated keys (starting points advanced by kernel)
int cuda_bech32_get_keys(GpuContext *ctx, uint8_t *keys, size_t size) {
    if (!ctx || !keys || size > ctx->keysSize) return -1;
    
    cudaError_t err = cudaMemcpy(keys, ctx->d_keys, size, cudaMemcpyDeviceToHost);
    return (err == cudaSuccess) ? 0 : -1;
}

// Calculate keys per launch based on configuration
uint64_t cuda_bech32_keys_per_launch(int numThreadGroups, int threadsPerGroup) {
    // Each thread processes STEP_SIZE keys
    // With endomorphism, each key gives 6 addresses
    // GRP_SIZE keys per step, STEP_SIZE/GRP_SIZE iterations
    return (uint64_t)numThreadGroups * threadsPerGroup * STEP_SIZE * 6;
}

} // extern "C"

 
 