/*
 * VanitySearch-POCX OpenCL Bech32 Kernel
 * Main kernel for bech32 vanity address search
 * 
 * Ported from VanitySearch GPUBech32.cu/h (CUDA) by Jean Luc PONS
 */

#ifndef BECH32_KERNEL_CL
#define BECH32_KERNEL_CL

// Note: group.cl, math.cl, hash.cl are concatenated before this file
// No includes needed

// Step size: number of keys processed per work-item per kernel launch
#ifndef STEP_SIZE
#define STEP_SIZE 16384
#endif

// Item size for match results (8 x uint32)
#define ITEM_SIZE32_BECH32 8

// ---------------------------------------------------------------------------------------
// Bech32 pattern checking
// ---------------------------------------------------------------------------------------

// Check if hash160 matches the bech32 pattern
// Pattern is pre-converted to 5-bit values
inline bool CheckBech32Pattern(
    __private uint *h,              // hash160 (5 x uint32)
    __global uchar *pattern5bit,    // Pattern as 5-bit values
    int patternLen                  // Length of pattern
) {
    if (patternLen == 0) return true;
    
    // Extract first 4 bytes and compute first 6 bech32 chars for quick reject
    uint h0 = h[0];
    uchar b0 = h0 & 0xFF;
    uchar b1 = (h0 >> 8) & 0xFF;
    uchar b2 = (h0 >> 16) & 0xFF;
    uchar b3 = (h0 >> 24) & 0xFF;
    
    // Combine to big-endian 32-bit for bit extraction
    uint data32 = ((uint)b0 << 24) | ((uint)b1 << 16) | ((uint)b2 << 8) | b3;
    
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
    uchar char5 = (data32 >> 7) & 0x1F;
    if (char5 != pattern5bit[4]) return false;
    if (patternLen == 5) return true;
    
    // Check sixth char
    uchar char6 = (data32 >> 2) & 0x1F;
    uchar expected6 = pattern5bit[5];
    if (char6 != expected6) return false;
    if (patternLen == 6) return true;
    
    // For patterns > 6 chars, continue with remaining bytes
    // We have 2 remaining bits from data32 in the lowest positions
    uint acc = data32 & 0x3;  // Remaining 2 bits from data32
    int bits = 2;
    int byteIdx = 4;  // Start from 5th byte (h[1])
    
    for (int i = 6; i < patternLen && i < 32; i++) {
        while (bits < 5 && byteIdx < 20) {
            // Extract byte from hash160
            // h[] is stored in little-endian (like CUDA)
            // byteIdx 4 = h[1] & 0xFF, byteIdx 5 = (h[1] >> 8) & 0xFF, etc.
            uint word = h[byteIdx >> 2];
            int shift = (byteIdx & 3) << 3;
            uchar b = (uchar)((word >> shift) & 0xFF);
            acc = (acc << 8) | (uint)b;
            bits += 8;
            byteIdx++;
        }
        bits -= 5;
        uchar extracted = (uchar)((acc >> bits) & 0x1F);
        
        if (extracted != pattern5bit[i]) {
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------------------
// Record a match
// ---------------------------------------------------------------------------------------

void CheckPointBech32(
    __private uint *h,
    int incr,
    int endo,
    __global uchar *pattern5bit,
    int patternLen,
    uint maxFound,
    __global uint *out,
    uint tid
) {
    if (!CheckBech32Pattern(h, pattern5bit, patternLen)) {
        return;
    }
    
    // Match found - atomically increment counter and store result
    uint pos = atomic_inc(out);
    if (pos < maxFound) {
        __global uint *entry = out + 1 + pos * ITEM_SIZE32_BECH32;
        entry[0] = tid;
        entry[1] = (uint)((incr << 16) | (endo & 0xFFFF));
        entry[2] = h[0];
        entry[3] = h[1];
        entry[4] = h[2];
        entry[5] = h[3];
        entry[6] = h[4];
        entry[7] = 0;  // Reserved
    }
}

// ---------------------------------------------------------------------------------------
// Main compute kernel for bech32 search
// ---------------------------------------------------------------------------------------

__kernel void bech32_search(
    __global ulong *startx,        // Starting X coordinates
    __global ulong *starty,        // Starting Y coordinates
    __global uchar *pattern5bit,   // Pattern in 5-bit values
    int patternLen,                // Pattern length
    uint maxFound,                 // Max matches to record
    __global uint *out             // Output: [count, match0, match1, ...]
) {
    uint tid = get_global_id(0);
    uint stride = get_global_size(0);
    
    // Local arrays for computation
    ulong dx[GRP_SIZE/2+1][4];
    ulong px[4], py[4], pyn[4];
    ulong sx[4], sy[4];
    ulong dy[4], _s[4], _p2[4];
    uint h1[5], h2[5];
    ulong pe1x[4], pe2x[4];
    
    // Load starting point
    Load256G(startx, tid, stride, sx);
    Load256G(starty, tid, stride, sy);
    Load256(px, sx);
    Load256(py, sy);
    
    // Process STEP_SIZE/GRP_SIZE groups
    for (uint j = 0; j < STEP_SIZE / GRP_SIZE; j++) {
        
        // Fill group with delta x values
        uint i;
        for (i = 0; i < HSIZE; i++) {
            ModSub256C(dx[i], Gx[i], sx);
        }
        ModSub256C(dx[i], Gx[i], sx);      // For the first point
        ModSub256C(dx[i+1], _2Gnx, sx);    // For next center point
        
        // Batch modular inverse using Montgomery's trick
        // dx[0..HSIZE+1] = dx[0..512] = 513 elements
        ModInvGrouped(dx, GRP_SIZE/2 + 1);
        
        // Check starting point (center of group)
        Load256(px, sx);
        GetHash160CompSym(px, h1, h2);
        CheckPointBech32(h1, j * GRP_SIZE + (GRP_SIZE / 2), 0, pattern5bit, patternLen, maxFound, out, tid);
        CheckPointBech32(h2, -(j * GRP_SIZE + (GRP_SIZE / 2)), 0, pattern5bit, patternLen, maxFound, out, tid);
        ModMultC(pe1x, px, _beta);
        GetHash160CompSym(pe1x, h1, h2);
        CheckPointBech32(h1, j * GRP_SIZE + (GRP_SIZE / 2), 1, pattern5bit, patternLen, maxFound, out, tid);
        CheckPointBech32(h2, -(j * GRP_SIZE + (GRP_SIZE / 2)), 1, pattern5bit, patternLen, maxFound, out, tid);
        ModMultC(pe2x, px, _beta2);
        GetHash160CompSym(pe2x, h1, h2);
        CheckPointBech32(h1, j * GRP_SIZE + (GRP_SIZE / 2), 2, pattern5bit, patternLen, maxFound, out, tid);
        CheckPointBech32(h2, -(j * GRP_SIZE + (GRP_SIZE / 2)), 2, pattern5bit, patternLen, maxFound, out, tid);
        
        // Negate py for symmetric checks
        ModNeg256(pyn, py);
        
        // Process points in group
        for (i = 0; i < HSIZE; i++) {
            // P = StartPoint + (i+1)*G
            Load256(px, sx);
            Load256(py, sy);
            ModSub256C(dy, Gy[i], py);
            
            ModMult(_s, dy, dx[i]);
            ModSqr(_p2, _s);
            
            ModSub256(px, _p2, px);
            ModSub256C_ip(px, Gx[i]);
            
            // Check this point with endomorphism
            GetHash160CompSym(px, h1, h2);
            CheckPointBech32(h1, j * GRP_SIZE + (GRP_SIZE / 2 + (i + 1)), 0, pattern5bit, patternLen, maxFound, out, tid);
            CheckPointBech32(h2, -(j * GRP_SIZE + (GRP_SIZE / 2 + (i + 1))), 0, pattern5bit, patternLen, maxFound, out, tid);
            ModMultC(pe1x, px, _beta);
            GetHash160CompSym(pe1x, h1, h2);
            CheckPointBech32(h1, j * GRP_SIZE + (GRP_SIZE / 2 + (i + 1)), 1, pattern5bit, patternLen, maxFound, out, tid);
            CheckPointBech32(h2, -(j * GRP_SIZE + (GRP_SIZE / 2 + (i + 1))), 1, pattern5bit, patternLen, maxFound, out, tid);
            ModMultC(pe2x, px, _beta2);
            GetHash160CompSym(pe2x, h1, h2);
            CheckPointBech32(h1, j * GRP_SIZE + (GRP_SIZE / 2 + (i + 1)), 2, pattern5bit, patternLen, maxFound, out, tid);
            CheckPointBech32(h2, -(j * GRP_SIZE + (GRP_SIZE / 2 + (i + 1))), 2, pattern5bit, patternLen, maxFound, out, tid);
            
            // P = StartPoint - (i+1)*G
            Load256(px, sx);
            ModSub256PC(dy, pyn, Gy[i]);
            
            ModMult(_s, dy, dx[i]);
            ModSqr(_p2, _s);
            
            ModSub256(px, _p2, px);
            ModSub256C_ip(px, Gx[i]);
            
            // Check this point with endomorphism
            GetHash160CompSym(px, h1, h2);
            CheckPointBech32(h1, j * GRP_SIZE + (GRP_SIZE / 2 - (i + 1)), 0, pattern5bit, patternLen, maxFound, out, tid);
            CheckPointBech32(h2, -(j * GRP_SIZE + (GRP_SIZE / 2 - (i + 1))), 0, pattern5bit, patternLen, maxFound, out, tid);
            ModMultC(pe1x, px, _beta);
            GetHash160CompSym(pe1x, h1, h2);
            CheckPointBech32(h1, j * GRP_SIZE + (GRP_SIZE / 2 - (i + 1)), 1, pattern5bit, patternLen, maxFound, out, tid);
            CheckPointBech32(h2, -(j * GRP_SIZE + (GRP_SIZE / 2 - (i + 1))), 1, pattern5bit, patternLen, maxFound, out, tid);
            ModMultC(pe2x, px, _beta2);
            GetHash160CompSym(pe2x, h1, h2);
            CheckPointBech32(h1, j * GRP_SIZE + (GRP_SIZE / 2 - (i + 1)), 2, pattern5bit, patternLen, maxFound, out, tid);
            CheckPointBech32(h2, -(j * GRP_SIZE + (GRP_SIZE / 2 - (i + 1))), 2, pattern5bit, patternLen, maxFound, out, tid);
        }
        
        // First point (startP - (GRP_SIZE/2)*G)
        Load256(px, sx);
        Load256(py, sy);
        ModNeg256C(dy, Gy[i]);
        ModSub256_ip(dy, py);
        
        ModMult(_s, dy, dx[i]);
        ModSqr(_p2, _s);
        
        ModSub256(px, _p2, px);
        ModSub256C_ip(px, Gx[i]);
        
        // Check this point with endomorphism
        GetHash160CompSym(px, h1, h2);
        CheckPointBech32(h1, j * GRP_SIZE + 0, 0, pattern5bit, patternLen, maxFound, out, tid);
        CheckPointBech32(h2, -(j * GRP_SIZE + 0), 0, pattern5bit, patternLen, maxFound, out, tid);
        ModMultC(pe1x, px, _beta);
        GetHash160CompSym(pe1x, h1, h2);
        CheckPointBech32(h1, j * GRP_SIZE + 0, 1, pattern5bit, patternLen, maxFound, out, tid);
        CheckPointBech32(h2, -(j * GRP_SIZE + 0), 1, pattern5bit, patternLen, maxFound, out, tid);
        ModMultC(pe2x, px, _beta2);
        GetHash160CompSym(pe2x, h1, h2);
        CheckPointBech32(h1, j * GRP_SIZE + 0, 2, pattern5bit, patternLen, maxFound, out, tid);
        CheckPointBech32(h2, -(j * GRP_SIZE + 0), 2, pattern5bit, patternLen, maxFound, out, tid);
        
        i++;
        
        // Next start point (startP + GRP_SIZE*G)
        Load256(px, sx);
        Load256(py, sy);
        ModSub256C(dy, _2Gny, py);
        
        ModMult(_s, dy, dx[i]);
        ModSqr(_p2, _s);
        
        ModSub256(px, _p2, px);
        ModSub256C_ip(px, _2Gnx);
        
        // Compute new Y
        ModSub256C(py, _2Gnx, px);
        ModMult_ip(py, _s);
        ModSub256C_ip(py, _2Gny);
        
        // Update start point for next iteration
        Load256(sx, px);
        Load256(sy, py);
    }
    
    // Store updated starting point for next kernel launch
    Store256G(startx, tid, stride, px);
    Store256G(starty, tid, stride, py);
}

#endif // BECH32_KERNEL_CL
