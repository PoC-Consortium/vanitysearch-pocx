/*
 * Bech32 GPU kernel for VanitySearch-POCX
 * Based on VanitySearch by Jean Luc PONS
 * 
 * For bech32 vanity search:
 * - Pattern matching on hash160 converted to 5-bit bech32 data
 * - No checksum needed during search (only for display)
 * - Pattern is searched after witness version (skip first char 'q')
 */

#ifndef GPUBECH32_H
#define GPUBECH32_H

// STEP_SIZE: number of keys processed per thread per kernel launch
// 16384 provides good balance
#ifndef STEP_SIZE
#define STEP_SIZE 16384
#endif

// Bech32 charset: qpzry9x8gf2tvdw0s3jn54khce6mua7l
// Each char maps to 0-31 (5 bits)
__device__ __constant__ char BECH32_CHARSET[32] = {
    'q', 'p', 'z', 'r', 'y', '9', 'x', '8',
    'g', 'f', '2', 't', 'v', 'd', 'w', '0',
    's', '3', 'j', 'n', '5', '4', 'k', 'h',
    'c', 'e', '6', 'm', 'u', 'a', '7', 'l'
};

// Reverse lookup: ASCII char -> 5-bit value (-1 for invalid)
__device__ __constant__ int8_t BECH32_REV[128] = {
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    15, -1, 10, 17, 21, 20, 26, 30,  7,  5, -1, -1, -1, -1, -1, -1,  // 0-9
    -1, 29, -1, 24, 13, 25,  9,  8, 23, -1, 18, 22, 31, 27, 19, -1,  // A-O
     1,  0,  3, 16, 11, 28, 12, 14,  6,  4,  2, -1, -1, -1, -1, -1,  // P-Z
    -1, 29, -1, 24, 13, 25,  9,  8, 23, -1, 18, 22, 31, 27, 19, -1,  // a-o
     1,  0,  3, 16, 11, 28, 12, 14,  6,  4,  2, -1, -1, -1, -1, -1,  // p-z
};

// Convert hash160 (20 bytes = 160 bits) to bech32 5-bit values (32 values)
// Output: 32 x 5-bit values = 160 bits
__device__ __forceinline__ void _Hash160ToBech32Data(uint32_t *h, uint8_t *out) {
    // h[0..4] contains hash160 in native byte order
    // We need to convert 160 bits -> 32 x 5-bit values
    
    // Extract bytes from hash160 (big-endian within each uint32)
    uint8_t bytes[20];
    
    // Note: hash160 from RIPEMD160 is in native format
    // Need to extract bytes properly
    bytes[0]  = (h[0] >> 0) & 0xFF;
    bytes[1]  = (h[0] >> 8) & 0xFF;
    bytes[2]  = (h[0] >> 16) & 0xFF;
    bytes[3]  = (h[0] >> 24) & 0xFF;
    bytes[4]  = (h[1] >> 0) & 0xFF;
    bytes[5]  = (h[1] >> 8) & 0xFF;
    bytes[6]  = (h[1] >> 16) & 0xFF;
    bytes[7]  = (h[1] >> 24) & 0xFF;
    bytes[8]  = (h[2] >> 0) & 0xFF;
    bytes[9]  = (h[2] >> 8) & 0xFF;
    bytes[10] = (h[2] >> 16) & 0xFF;
    bytes[11] = (h[2] >> 24) & 0xFF;
    bytes[12] = (h[3] >> 0) & 0xFF;
    bytes[13] = (h[3] >> 8) & 0xFF;
    bytes[14] = (h[3] >> 16) & 0xFF;
    bytes[15] = (h[3] >> 24) & 0xFF;
    bytes[16] = (h[4] >> 0) & 0xFF;
    bytes[17] = (h[4] >> 8) & 0xFF;
    bytes[18] = (h[4] >> 16) & 0xFF;
    bytes[19] = (h[4] >> 24) & 0xFF;
    
    // Convert 8-bit to 5-bit (160 bits = 32 x 5-bit values)
    uint32_t acc = 0;
    int bits = 0;
    int outIdx = 0;
    
    #pragma unroll
    for (int i = 0; i < 20; i++) {
        acc = (acc << 8) | bytes[i];
        bits += 8;
        while (bits >= 5 && outIdx < 32) {
            bits -= 5;
            out[outIdx++] = (acc >> bits) & 0x1F;
        }
    }
}

// Fast pattern check for bech32 - optimized inline version
// Pattern is pre-converted to 5-bit values on host
__device__ __forceinline__ bool _CheckBech32Pattern(
    uint32_t *h,           // hash160 (5 x uint32)
    uint8_t *pattern5bit,  // Pattern as 5-bit values
    int patternLen         // Length of pattern in 5-bit values
) {
    if (patternLen == 0) return true;
    
    // Extract first 4 bytes and compute first 6 bech32 chars for quick reject
    uint32_t h0 = h[0];
    uint8_t b0 = h0 & 0xFF;
    uint8_t b1 = (h0 >> 8) & 0xFF;
    uint8_t b2 = (h0 >> 16) & 0xFF;
    uint8_t b3 = (h0 >> 24) & 0xFF;
    
    // Combine to big-endian 32-bit for bit extraction
    uint32_t data32 = ((uint32_t)b0 << 24) | ((uint32_t)b1 << 16) | ((uint32_t)b2 << 8) | b3;
    
    // Check first char (5 bits from top)
    if (((data32 >> 27) & 0x1F) != pattern5bit[0]) return false;
    if (patternLen == 1) return true;
    
    // Check second char
    if (((data32 >> 22) & 0x1F) != pattern5bit[1]) return false;
    if (patternLen == 2) return true;
    
    // Check third char
    if (((data32 >> 17) & 0x1F) != pattern5bit[2]) return false;
    if (patternLen == 3) return true;
    
    // Check fourth char
    if (((data32 >> 12) & 0x1F) != pattern5bit[3]) return false;
    if (patternLen == 4) return true;
    
    // Check fifth char
    if (((data32 >> 7) & 0x1F) != pattern5bit[4]) return false;
    if (patternLen == 5) return true;
    
    // Check sixth char
    if (((data32 >> 2) & 0x1F) != pattern5bit[5]) return false;
    if (patternLen == 6) return true;
    
    // For patterns > 6 chars, continue with remaining bytes
    uint32_t acc = data32 & 0x3;  // Remaining 2 bits from data32
    int bits = 2;
    int byteIdx = 4;  // Start from 5th byte (h[1])
    
    for (int i = 6; i < patternLen && i < 32; i++) {
        while (bits < 5 && byteIdx < 20) {
            uint32_t word = h[byteIdx >> 2];
            int shift = (byteIdx & 3) << 3;
            uint8_t b = (word >> shift) & 0xFF;
            acc = (acc << 8) | b;
            bits += 8;
            byteIdx++;
        }
        bits -= 5;
        if (((acc >> bits) & 0x1F) != pattern5bit[i]) {
            return false;
        }
    }
    return true;
}

// Item size for results (matching VanitySearch format)
#define ITEM_SIZE32_BECH32 8

// Check point for bech32 matching
__device__ __noinline__ void CheckPointBech32(
    uint32_t *_h,
    int32_t incr,
    int32_t endo,
    uint8_t *pattern5bit,
    int patternLen,
    uint32_t maxFound,
    uint32_t *out
) {
    uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    
    // Check if hash160 matches pattern
    if (!_CheckBech32Pattern(_h, pattern5bit, patternLen)) {
        return;
    }
    
    // Match found - add to output
    uint32_t pos = atomicAdd(out, 1);
    if (pos < maxFound) {
        out[pos * ITEM_SIZE32_BECH32 + 1] = tid;
        out[pos * ITEM_SIZE32_BECH32 + 2] = (uint32_t)(incr << 16) | (uint32_t)(endo);
        out[pos * ITEM_SIZE32_BECH32 + 3] = _h[0];
        out[pos * ITEM_SIZE32_BECH32 + 4] = _h[1];
        out[pos * ITEM_SIZE32_BECH32 + 5] = _h[2];
        out[pos * ITEM_SIZE32_BECH32 + 6] = _h[3];
        out[pos * ITEM_SIZE32_BECH32 + 7] = _h[4];
    }
}

// Macro for checking bech32 point (similar to CHECK_P2PKH_POINT)
#define CHECK_BECH32_POINT(_incr) {                                                    \
    _GetHash160CompSym(px, (uint8_t *)h1, (uint8_t *)h2);                              \
    CheckPointBech32(h1, (_incr), 0, pattern5bit, patternLen, maxFound, out);          \
    CheckPointBech32(h2, -(_incr), 0, pattern5bit, patternLen, maxFound, out);         \
    _ModMult(pe1x, px, _beta);                                                         \
    _GetHash160CompSym(pe1x, (uint8_t *)h1, (uint8_t *)h2);                            \
    CheckPointBech32(h1, (_incr), 1, pattern5bit, patternLen, maxFound, out);          \
    CheckPointBech32(h2, -(_incr), 1, pattern5bit, patternLen, maxFound, out);         \
    _ModMult(pe2x, px, _beta2);                                                        \
    _GetHash160CompSym(pe2x, (uint8_t *)h1, (uint8_t *)h2);                            \
    CheckPointBech32(h1, (_incr), 2, pattern5bit, patternLen, maxFound, out);          \
    CheckPointBech32(h2, -(_incr), 2, pattern5bit, patternLen, maxFound, out);         \
}

// Main kernel for bech32 compressed key search
// Adapted from ComputeKeysComp in GPUCompute.h
__device__ void ComputeKeysBech32(
    uint64_t *startx,
    uint64_t *starty,
    uint8_t *pattern5bit,
    int patternLen,
    uint32_t maxFound,
    uint32_t *out
) {
    uint64_t dx[GRP_SIZE/2+1][4];
    uint64_t px[4];
    uint64_t py[4];
    uint64_t pyn[4];
    uint64_t sx[4];
    uint64_t sy[4];
    uint64_t dy[4];
    uint64_t _s[4];
    uint64_t _p2[4];
    uint32_t h1[5];
    uint32_t h2[5];
    uint64_t pe1x[4];
    uint64_t pe2x[4];

    // Load starting key
    __syncthreads();
    Load256A(sx, startx);
    Load256A(sy, starty);
    Load256(px, sx);
    Load256(py, sy);

    for (uint32_t j = 0; j < STEP_SIZE / GRP_SIZE; j++) {
        // Fill group with delta x
        uint32_t i;
        for (i = 0; i < HSIZE; i++)
            ModSub256(dx[i], Gx[i], sx);
        ModSub256(dx[i], Gx[i], sx);     // For the first point
        ModSub256(dx[i+1], _2Gnx, sx);   // For the next center point

        // Compute modular inverse (batch)
        _ModInvGrouped(dx);

        // Check starting point (center of group)
        CHECK_BECH32_POINT(j * GRP_SIZE + (GRP_SIZE / 2));

        ModNeg256(pyn, py);

        for (i = 0; i < HSIZE; i++) {
            __syncthreads();
            // P = StartPoint + i*G
            Load256(px, sx);
            Load256(py, sy);
            ModSub256(dy, Gy[i], py);

            _ModMult(_s, dy, dx[i]);
            _ModSqr(_p2, _s);

            ModSub256(px, _p2, px);
            ModSub256(px, Gx[i]);

            CHECK_BECH32_POINT(j * GRP_SIZE + (GRP_SIZE / 2 + (i + 1)));

            __syncthreads();
            // P = StartPoint - i*G
            Load256(px, sx);
            ModSub256(dy, pyn, Gy[i]);

            _ModMult(_s, dy, dx[i]);
            _ModSqr(_p2, _s);

            ModSub256(px, _p2, px);
            ModSub256(px, Gx[i]);

            CHECK_BECH32_POINT(j * GRP_SIZE + (GRP_SIZE / 2 - (i + 1)));
        }

        __syncthreads();
        // First point (startP - (GRP_SIZE/2)*G)
        Load256(px, sx);
        Load256(py, sy);
        ModNeg256(dy, Gy[i]);
        ModSub256(dy, py);

        _ModMult(_s, dy, dx[i]);
        _ModSqr(_p2, _s);

        ModSub256(px, _p2, px);
        ModSub256(px, Gx[i]);

        CHECK_BECH32_POINT(j * GRP_SIZE + 0);

        i++;

        __syncthreads();
        // Next start point (startP + GRP_SIZE*G)
        Load256(px, sx);
        Load256(py, sy);
        ModSub256(dy, _2Gny, py);

        _ModMult(_s, dy, dx[i]);
        _ModSqr(_p2, _s);

        ModSub256(px, _p2, px);
        ModSub256(px, _2Gnx);

        ModSub256(py, _2Gnx, px);
        _ModMult(py, _s);
        ModSub256(py, _2Gny);
    }

    // Update starting point
    __syncthreads();
    Store256A(startx, px);
    Store256A(starty, py);
}

#endif // GPUBECH32_H
