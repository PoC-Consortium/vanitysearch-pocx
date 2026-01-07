/*
 * VanitySearch-POCX OpenCL Hash Library
 * SHA256 and RIPEMD160 for OpenCL
 * 
 * Ported from VanitySearch GPUHash.h (CUDA) by Jean Luc PONS
 */

// ---------------------------------------------------------------------------------------
// SHA256 Constants
// ---------------------------------------------------------------------------------------

__constant uint K[64] = {
    0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5,
    0x3956C25B, 0x59F111F1, 0x923F82A4, 0xAB1C5ED5,
    0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3,
    0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7, 0xC19BF174,
    0xE49B69C1, 0xEFBE4786, 0x0FC19DC6, 0x240CA1CC,
    0x2DE92C6F, 0x4A7484AA, 0x5CB0A9DC, 0x76F988DA,
    0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7,
    0xC6E00BF3, 0xD5A79147, 0x06CA6351, 0x14292967,
    0x27B70A85, 0x2E1B2138, 0x4D2C6DFC, 0x53380D13,
    0x650A7354, 0x766A0ABB, 0x81C2C92E, 0x92722C85,
    0xA2BFE8A1, 0xA81A664B, 0xC24B8B70, 0xC76C51A3,
    0xD192E819, 0xD6990624, 0xF40E3585, 0x106AA070,
    0x19A4C116, 0x1E376C08, 0x2748774C, 0x34B0BCB5,
    0x391C0CB3, 0x4ED8AA4A, 0x5B9CCA4F, 0x682E6FF3,
    0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208,
    0x90BEFFFA, 0xA4506CEB, 0xBEF9A3F7, 0xC67178F2,
};

__constant uint SHA256_IV[8] = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
};

// ---------------------------------------------------------------------------------------
// SHA256 Helper Functions
// ---------------------------------------------------------------------------------------

#define ROR32(x, n) (((x) >> (n)) | ((x) << (32 - (n))))

#define SHA256_S0(x) (ROR32(x, 2) ^ ROR32(x, 13) ^ ROR32(x, 22))
#define SHA256_S1(x) (ROR32(x, 6) ^ ROR32(x, 11) ^ ROR32(x, 25))
#define SHA256_s0(x) (ROR32(x, 7) ^ ROR32(x, 18) ^ ((x) >> 3))
#define SHA256_s1(x) (ROR32(x, 17) ^ ROR32(x, 19) ^ ((x) >> 10))

#define SHA256_Maj(x, y, z) (((x) & (y)) | ((z) & ((x) | (y))))
#define SHA256_Ch(x, y, z) ((z) ^ ((x) & ((y) ^ (z))))

// SHA256 round
#define SHA256_Round(a, b, c, d, e, f, g, h, k, w) { \
    uint t1 = (h) + SHA256_S1(e) + SHA256_Ch(e, f, g) + (k) + (w); \
    uint t2 = SHA256_S0(a) + SHA256_Maj(a, b, c); \
    (d) += t1; \
    (h) = t1 + t2; \
}

// Byte swap (big-endian to little-endian)
inline uint bswap32(uint v) {
    return ((v >> 24) & 0xFF) | ((v >> 8) & 0xFF00) | ((v << 8) & 0xFF0000) | ((v << 24) & 0xFF000000);
}

// ---------------------------------------------------------------------------------------
// SHA256 Transform
// ---------------------------------------------------------------------------------------

inline void SHA256Initialize(__private uint *s) {
    s[0] = 0x6a09e667;
    s[1] = 0xbb67ae85;
    s[2] = 0x3c6ef372;
    s[3] = 0xa54ff53a;
    s[4] = 0x510e527f;
    s[5] = 0x9b05688c;
    s[6] = 0x1f83d9ab;
    s[7] = 0x5be0cd19;
}

void SHA256Transform(__private uint *s, __private uint *w) {
    uint a = s[0], b = s[1], c = s[2], d = s[3];
    uint e = s[4], f = s[5], g = s[6], h = s[7];
    
    // Rounds 0-15
    SHA256_Round(a, b, c, d, e, f, g, h, K[0], w[0]);
    SHA256_Round(h, a, b, c, d, e, f, g, K[1], w[1]);
    SHA256_Round(g, h, a, b, c, d, e, f, K[2], w[2]);
    SHA256_Round(f, g, h, a, b, c, d, e, K[3], w[3]);
    SHA256_Round(e, f, g, h, a, b, c, d, K[4], w[4]);
    SHA256_Round(d, e, f, g, h, a, b, c, K[5], w[5]);
    SHA256_Round(c, d, e, f, g, h, a, b, K[6], w[6]);
    SHA256_Round(b, c, d, e, f, g, h, a, K[7], w[7]);
    SHA256_Round(a, b, c, d, e, f, g, h, K[8], w[8]);
    SHA256_Round(h, a, b, c, d, e, f, g, K[9], w[9]);
    SHA256_Round(g, h, a, b, c, d, e, f, K[10], w[10]);
    SHA256_Round(f, g, h, a, b, c, d, e, K[11], w[11]);
    SHA256_Round(e, f, g, h, a, b, c, d, K[12], w[12]);
    SHA256_Round(d, e, f, g, h, a, b, c, K[13], w[13]);
    SHA256_Round(c, d, e, f, g, h, a, b, K[14], w[14]);
    SHA256_Round(b, c, d, e, f, g, h, a, K[15], w[15]);
    
    // Message schedule expansion and rounds 16-63
    #pragma unroll
    for (int i = 16; i < 64; i += 16) {
        w[0] += SHA256_s1(w[14]) + w[9] + SHA256_s0(w[1]);
        w[1] += SHA256_s1(w[15]) + w[10] + SHA256_s0(w[2]);
        w[2] += SHA256_s1(w[0]) + w[11] + SHA256_s0(w[3]);
        w[3] += SHA256_s1(w[1]) + w[12] + SHA256_s0(w[4]);
        w[4] += SHA256_s1(w[2]) + w[13] + SHA256_s0(w[5]);
        w[5] += SHA256_s1(w[3]) + w[14] + SHA256_s0(w[6]);
        w[6] += SHA256_s1(w[4]) + w[15] + SHA256_s0(w[7]);
        w[7] += SHA256_s1(w[5]) + w[0] + SHA256_s0(w[8]);
        w[8] += SHA256_s1(w[6]) + w[1] + SHA256_s0(w[9]);
        w[9] += SHA256_s1(w[7]) + w[2] + SHA256_s0(w[10]);
        w[10] += SHA256_s1(w[8]) + w[3] + SHA256_s0(w[11]);
        w[11] += SHA256_s1(w[9]) + w[4] + SHA256_s0(w[12]);
        w[12] += SHA256_s1(w[10]) + w[5] + SHA256_s0(w[13]);
        w[13] += SHA256_s1(w[11]) + w[6] + SHA256_s0(w[14]);
        w[14] += SHA256_s1(w[12]) + w[7] + SHA256_s0(w[15]);
        w[15] += SHA256_s1(w[13]) + w[8] + SHA256_s0(w[0]);
        
        SHA256_Round(a, b, c, d, e, f, g, h, K[i+0], w[0]);
        SHA256_Round(h, a, b, c, d, e, f, g, K[i+1], w[1]);
        SHA256_Round(g, h, a, b, c, d, e, f, K[i+2], w[2]);
        SHA256_Round(f, g, h, a, b, c, d, e, K[i+3], w[3]);
        SHA256_Round(e, f, g, h, a, b, c, d, K[i+4], w[4]);
        SHA256_Round(d, e, f, g, h, a, b, c, K[i+5], w[5]);
        SHA256_Round(c, d, e, f, g, h, a, b, K[i+6], w[6]);
        SHA256_Round(b, c, d, e, f, g, h, a, K[i+7], w[7]);
        SHA256_Round(a, b, c, d, e, f, g, h, K[i+8], w[8]);
        SHA256_Round(h, a, b, c, d, e, f, g, K[i+9], w[9]);
        SHA256_Round(g, h, a, b, c, d, e, f, K[i+10], w[10]);
        SHA256_Round(f, g, h, a, b, c, d, e, K[i+11], w[11]);
        SHA256_Round(e, f, g, h, a, b, c, d, K[i+12], w[12]);
        SHA256_Round(d, e, f, g, h, a, b, c, K[i+13], w[13]);
        SHA256_Round(c, d, e, f, g, h, a, b, K[i+14], w[14]);
        SHA256_Round(b, c, d, e, f, g, h, a, K[i+15], w[15]);
    }
    
    s[0] += a; s[1] += b; s[2] += c; s[3] += d;
    s[4] += e; s[5] += f; s[6] += g; s[7] += h;
}

// ---------------------------------------------------------------------------------------
// RIPEMD160 Constants
// ---------------------------------------------------------------------------------------

#define ROL32(x, n) (((x) << (n)) | ((x) >> (32 - (n))))

#define RIPEMD160_f1(x, y, z) ((x) ^ (y) ^ (z))
#define RIPEMD160_f2(x, y, z) (((x) & (y)) | (~(x) & (z)))
#define RIPEMD160_f3(x, y, z) (((x) | ~(y)) ^ (z))
#define RIPEMD160_f4(x, y, z) (((x) & (z)) | ((y) & ~(z)))
#define RIPEMD160_f5(x, y, z) ((x) ^ ((y) | ~(z)))

#define RIPEMD160_Round(a, b, c, d, e, f, x, k, r) { \
    uint u = (a) + (f) + (x) + (k); \
    (a) = ROL32(u, (r)) + (e); \
    (c) = ROL32((c), 10); \
}

#define R11(a, b, c, d, e, x, r) RIPEMD160_Round(a, b, c, d, e, RIPEMD160_f1(b, c, d), x, 0x00000000, r)
#define R21(a, b, c, d, e, x, r) RIPEMD160_Round(a, b, c, d, e, RIPEMD160_f2(b, c, d), x, 0x5A827999, r)
#define R31(a, b, c, d, e, x, r) RIPEMD160_Round(a, b, c, d, e, RIPEMD160_f3(b, c, d), x, 0x6ED9EBA1, r)
#define R41(a, b, c, d, e, x, r) RIPEMD160_Round(a, b, c, d, e, RIPEMD160_f4(b, c, d), x, 0x8F1BBCDC, r)
#define R51(a, b, c, d, e, x, r) RIPEMD160_Round(a, b, c, d, e, RIPEMD160_f5(b, c, d), x, 0xA953FD4E, r)

#define R12(a, b, c, d, e, x, r) RIPEMD160_Round(a, b, c, d, e, RIPEMD160_f5(b, c, d), x, 0x50A28BE6, r)
#define R22(a, b, c, d, e, x, r) RIPEMD160_Round(a, b, c, d, e, RIPEMD160_f4(b, c, d), x, 0x5C4DD124, r)
#define R32(a, b, c, d, e, x, r) RIPEMD160_Round(a, b, c, d, e, RIPEMD160_f3(b, c, d), x, 0x6D703EF3, r)
#define R42(a, b, c, d, e, x, r) RIPEMD160_Round(a, b, c, d, e, RIPEMD160_f2(b, c, d), x, 0x7A6D76E9, r)
#define R52(a, b, c, d, e, x, r) RIPEMD160_Round(a, b, c, d, e, RIPEMD160_f1(b, c, d), x, 0x00000000, r)

// ---------------------------------------------------------------------------------------
// RIPEMD160 Transform
// ---------------------------------------------------------------------------------------

inline void RIPEMD160Initialize(__private uint *s) {
    s[0] = 0x67452301;
    s[1] = 0xEFCDAB89;
    s[2] = 0x98BADCFE;
    s[3] = 0x10325476;
    s[4] = 0xC3D2E1F0;
}

void RIPEMD160Transform(__private uint *s, __private uint *w) {
    uint a1 = s[0], b1 = s[1], c1 = s[2], d1 = s[3], e1 = s[4];
    uint a2 = a1, b2 = b1, c2 = c1, d2 = d1, e2 = e1;
    
    // Left rounds
    R11(a1, b1, c1, d1, e1, w[0], 11);
    R12(a2, b2, c2, d2, e2, w[5], 8);
    R11(e1, a1, b1, c1, d1, w[1], 14);
    R12(e2, a2, b2, c2, d2, w[14], 9);
    R11(d1, e1, a1, b1, c1, w[2], 15);
    R12(d2, e2, a2, b2, c2, w[7], 9);
    R11(c1, d1, e1, a1, b1, w[3], 12);
    R12(c2, d2, e2, a2, b2, w[0], 11);
    R11(b1, c1, d1, e1, a1, w[4], 5);
    R12(b2, c2, d2, e2, a2, w[9], 13);
    R11(a1, b1, c1, d1, e1, w[5], 8);
    R12(a2, b2, c2, d2, e2, w[2], 15);
    R11(e1, a1, b1, c1, d1, w[6], 7);
    R12(e2, a2, b2, c2, d2, w[11], 15);
    R11(d1, e1, a1, b1, c1, w[7], 9);
    R12(d2, e2, a2, b2, c2, w[4], 5);
    R11(c1, d1, e1, a1, b1, w[8], 11);
    R12(c2, d2, e2, a2, b2, w[13], 7);
    R11(b1, c1, d1, e1, a1, w[9], 13);
    R12(b2, c2, d2, e2, a2, w[6], 7);
    R11(a1, b1, c1, d1, e1, w[10], 14);
    R12(a2, b2, c2, d2, e2, w[15], 8);
    R11(e1, a1, b1, c1, d1, w[11], 15);
    R12(e2, a2, b2, c2, d2, w[8], 11);
    R11(d1, e1, a1, b1, c1, w[12], 6);
    R12(d2, e2, a2, b2, c2, w[1], 14);
    R11(c1, d1, e1, a1, b1, w[13], 7);
    R12(c2, d2, e2, a2, b2, w[10], 14);
    R11(b1, c1, d1, e1, a1, w[14], 9);
    R12(b2, c2, d2, e2, a2, w[3], 12);
    R11(a1, b1, c1, d1, e1, w[15], 8);
    R12(a2, b2, c2, d2, e2, w[12], 6);
    
    R21(e1, a1, b1, c1, d1, w[7], 7);
    R22(e2, a2, b2, c2, d2, w[6], 9);
    R21(d1, e1, a1, b1, c1, w[4], 6);
    R22(d2, e2, a2, b2, c2, w[11], 13);
    R21(c1, d1, e1, a1, b1, w[13], 8);
    R22(c2, d2, e2, a2, b2, w[3], 15);
    R21(b1, c1, d1, e1, a1, w[1], 13);
    R22(b2, c2, d2, e2, a2, w[7], 7);
    R21(a1, b1, c1, d1, e1, w[10], 11);
    R22(a2, b2, c2, d2, e2, w[0], 12);
    R21(e1, a1, b1, c1, d1, w[6], 9);
    R22(e2, a2, b2, c2, d2, w[13], 8);
    R21(d1, e1, a1, b1, c1, w[15], 7);
    R22(d2, e2, a2, b2, c2, w[5], 9);
    R21(c1, d1, e1, a1, b1, w[3], 15);
    R22(c2, d2, e2, a2, b2, w[10], 11);
    R21(b1, c1, d1, e1, a1, w[12], 7);
    R22(b2, c2, d2, e2, a2, w[14], 7);
    R21(a1, b1, c1, d1, e1, w[0], 12);
    R22(a2, b2, c2, d2, e2, w[15], 7);
    R21(e1, a1, b1, c1, d1, w[9], 15);
    R22(e2, a2, b2, c2, d2, w[8], 12);
    R21(d1, e1, a1, b1, c1, w[5], 9);
    R22(d2, e2, a2, b2, c2, w[12], 7);
    R21(c1, d1, e1, a1, b1, w[2], 11);
    R22(c2, d2, e2, a2, b2, w[4], 6);
    R21(b1, c1, d1, e1, a1, w[14], 7);
    R22(b2, c2, d2, e2, a2, w[9], 15);
    R21(a1, b1, c1, d1, e1, w[11], 13);
    R22(a2, b2, c2, d2, e2, w[1], 13);
    R21(e1, a1, b1, c1, d1, w[8], 12);
    R22(e2, a2, b2, c2, d2, w[2], 11);
    
    R31(d1, e1, a1, b1, c1, w[3], 11);
    R32(d2, e2, a2, b2, c2, w[15], 9);
    R31(c1, d1, e1, a1, b1, w[10], 13);
    R32(c2, d2, e2, a2, b2, w[5], 7);
    R31(b1, c1, d1, e1, a1, w[14], 6);
    R32(b2, c2, d2, e2, a2, w[1], 15);
    R31(a1, b1, c1, d1, e1, w[4], 7);
    R32(a2, b2, c2, d2, e2, w[3], 11);
    R31(e1, a1, b1, c1, d1, w[9], 14);
    R32(e2, a2, b2, c2, d2, w[7], 8);
    R31(d1, e1, a1, b1, c1, w[15], 9);
    R32(d2, e2, a2, b2, c2, w[14], 6);
    R31(c1, d1, e1, a1, b1, w[8], 13);
    R32(c2, d2, e2, a2, b2, w[6], 6);
    R31(b1, c1, d1, e1, a1, w[1], 15);
    R32(b2, c2, d2, e2, a2, w[9], 14);
    R31(a1, b1, c1, d1, e1, w[2], 14);
    R32(a2, b2, c2, d2, e2, w[11], 12);
    R31(e1, a1, b1, c1, d1, w[7], 8);
    R32(e2, a2, b2, c2, d2, w[8], 13);
    R31(d1, e1, a1, b1, c1, w[0], 13);
    R32(d2, e2, a2, b2, c2, w[12], 5);
    R31(c1, d1, e1, a1, b1, w[6], 6);
    R32(c2, d2, e2, a2, b2, w[2], 14);
    R31(b1, c1, d1, e1, a1, w[13], 5);
    R32(b2, c2, d2, e2, a2, w[10], 13);
    R31(a1, b1, c1, d1, e1, w[11], 12);
    R32(a2, b2, c2, d2, e2, w[0], 13);
    R31(e1, a1, b1, c1, d1, w[5], 7);
    R32(e2, a2, b2, c2, d2, w[4], 7);
    R31(d1, e1, a1, b1, c1, w[12], 5);
    R32(d2, e2, a2, b2, c2, w[13], 5);
    
    R41(c1, d1, e1, a1, b1, w[1], 11);
    R42(c2, d2, e2, a2, b2, w[8], 15);
    R41(b1, c1, d1, e1, a1, w[9], 12);
    R42(b2, c2, d2, e2, a2, w[6], 5);
    R41(a1, b1, c1, d1, e1, w[11], 14);
    R42(a2, b2, c2, d2, e2, w[4], 8);
    R41(e1, a1, b1, c1, d1, w[10], 15);
    R42(e2, a2, b2, c2, d2, w[1], 11);
    R41(d1, e1, a1, b1, c1, w[0], 14);
    R42(d2, e2, a2, b2, c2, w[3], 14);
    R41(c1, d1, e1, a1, b1, w[8], 15);
    R42(c2, d2, e2, a2, b2, w[11], 14);
    R41(b1, c1, d1, e1, a1, w[12], 9);
    R42(b2, c2, d2, e2, a2, w[15], 6);
    R41(a1, b1, c1, d1, e1, w[4], 8);
    R42(a2, b2, c2, d2, e2, w[0], 14);
    R41(e1, a1, b1, c1, d1, w[13], 9);
    R42(e2, a2, b2, c2, d2, w[5], 6);
    R41(d1, e1, a1, b1, c1, w[3], 14);
    R42(d2, e2, a2, b2, c2, w[12], 9);
    R41(c1, d1, e1, a1, b1, w[7], 5);
    R42(c2, d2, e2, a2, b2, w[2], 12);
    R41(b1, c1, d1, e1, a1, w[15], 6);
    R42(b2, c2, d2, e2, a2, w[13], 9);
    R41(a1, b1, c1, d1, e1, w[14], 8);
    R42(a2, b2, c2, d2, e2, w[9], 12);
    R41(e1, a1, b1, c1, d1, w[5], 6);
    R42(e2, a2, b2, c2, d2, w[7], 5);
    R41(d1, e1, a1, b1, c1, w[6], 5);
    R42(d2, e2, a2, b2, c2, w[10], 15);
    R41(c1, d1, e1, a1, b1, w[2], 12);
    R42(c2, d2, e2, a2, b2, w[14], 8);
    
    R51(b1, c1, d1, e1, a1, w[4], 9);
    R52(b2, c2, d2, e2, a2, w[12], 8);
    R51(a1, b1, c1, d1, e1, w[0], 15);
    R52(a2, b2, c2, d2, e2, w[15], 5);
    R51(e1, a1, b1, c1, d1, w[5], 5);
    R52(e2, a2, b2, c2, d2, w[10], 12);
    R51(d1, e1, a1, b1, c1, w[9], 11);
    R52(d2, e2, a2, b2, c2, w[4], 9);
    R51(c1, d1, e1, a1, b1, w[7], 6);
    R52(c2, d2, e2, a2, b2, w[1], 12);
    R51(b1, c1, d1, e1, a1, w[12], 8);
    R52(b2, c2, d2, e2, a2, w[5], 5);
    R51(a1, b1, c1, d1, e1, w[2], 13);
    R52(a2, b2, c2, d2, e2, w[8], 14);
    R51(e1, a1, b1, c1, d1, w[10], 12);
    R52(e2, a2, b2, c2, d2, w[7], 6);
    R51(d1, e1, a1, b1, c1, w[14], 5);
    R52(d2, e2, a2, b2, c2, w[6], 8);
    R51(c1, d1, e1, a1, b1, w[1], 12);
    R52(c2, d2, e2, a2, b2, w[2], 13);
    R51(b1, c1, d1, e1, a1, w[3], 13);
    R52(b2, c2, d2, e2, a2, w[13], 6);
    R51(a1, b1, c1, d1, e1, w[8], 14);
    R52(a2, b2, c2, d2, e2, w[14], 5);
    R51(e1, a1, b1, c1, d1, w[11], 11);
    R52(e2, a2, b2, c2, d2, w[0], 15);
    R51(d1, e1, a1, b1, c1, w[6], 8);
    R52(d2, e2, a2, b2, c2, w[3], 13);
    R51(c1, d1, e1, a1, b1, w[15], 5);
    R52(c2, d2, e2, a2, b2, w[9], 11);
    R51(b1, c1, d1, e1, a1, w[13], 6);
    R52(b2, c2, d2, e2, a2, w[11], 11);
    
    uint t = s[0];
    s[0] = s[1] + c1 + d2;
    s[1] = s[2] + d1 + e2;
    s[2] = s[3] + e1 + a2;
    s[3] = s[4] + a1 + b2;
    s[4] = t + b1 + c2;
}

// ---------------------------------------------------------------------------------------
// Hash160: SHA256 + RIPEMD160 for compressed public key
// ---------------------------------------------------------------------------------------

void GetHash160Comp(__private ulong *x, uint isOdd, __private uint *hash) {
    uint publicKeyBytes[16];
    uint s[16];
    
    // Build compressed public key (33 bytes, padded to 64 bytes)
    // Format: [02/03][X coordinate 32 bytes][padding]
    uint *x32 = (uint *)x;
    
    // Emulate CUDA's __byte_perm for correct byte ordering
    publicKeyBytes[0] = ((x32[7] >> 8) & 0xFFFFFF) | ((0x02 + isOdd) << 24);
    publicKeyBytes[1] = ((x32[6] >> 8) & 0xFFFFFF) | ((x32[7] & 0xFF) << 24);
    publicKeyBytes[2] = ((x32[5] >> 8) & 0xFFFFFF) | ((x32[6] & 0xFF) << 24);
    publicKeyBytes[3] = ((x32[4] >> 8) & 0xFFFFFF) | ((x32[5] & 0xFF) << 24);
    publicKeyBytes[4] = ((x32[3] >> 8) & 0xFFFFFF) | ((x32[4] & 0xFF) << 24);
    publicKeyBytes[5] = ((x32[2] >> 8) & 0xFFFFFF) | ((x32[3] & 0xFF) << 24);
    publicKeyBytes[6] = ((x32[1] >> 8) & 0xFFFFFF) | ((x32[2] & 0xFF) << 24);
    publicKeyBytes[7] = ((x32[0] >> 8) & 0xFFFFFF) | ((x32[1] & 0xFF) << 24);
    publicKeyBytes[8] = 0x00800000 | ((x32[0] & 0xFF) << 24);  // Padding start (0x80 in correct position)
    publicKeyBytes[9] = 0;
    publicKeyBytes[10] = 0;
    publicKeyBytes[11] = 0;
    publicKeyBytes[12] = 0;
    publicKeyBytes[13] = 0;
    publicKeyBytes[14] = 0;
    publicKeyBytes[15] = 0x108;  // Length in bits (33 * 8 = 264)
    
    // SHA256
    SHA256Initialize(s);
    SHA256Transform(s, publicKeyBytes);
    
    // Byte swap result
    #pragma unroll
    for (int i = 0; i < 8; i++)
        s[i] = bswap32(s[i]);
    
    // Prepare for RIPEMD160
    s[8] = 0x80;
    s[9] = 0;
    s[10] = 0;
    s[11] = 0;
    s[12] = 0;
    s[13] = 0;
    s[14] = 0x100;  // 256 bits
    s[15] = 0;
    
    // RIPEMD160
    RIPEMD160Initialize(hash);
    RIPEMD160Transform(hash, s);
}

// Compute hash160 for both even and odd Y (symmetric point optimization)
void GetHash160CompSym(__private ulong *x, __private uint *h1, __private uint *h2) {
    uint publicKeyBytes[16];
    uint publicKeyBytes2[16];
    uint s[16];
    
    uint *x32 = (uint *)x;
    
    // Even (02 prefix)
    // Emulate CUDA's __byte_perm(x32[7], 0x02, 0x4321) = {x[1], x[2], x[3], 0x02}
    publicKeyBytes[0] = ((x32[7] >> 8) & 0xFFFFFF) | 0x02000000;
    // Emulate CUDA's __byte_perm(x32[7], x32[6], 0x0765) = {y[1], y[2], y[3], x[0]}
    publicKeyBytes[1] = ((x32[6] >> 8) & 0xFFFFFF) | ((x32[7] & 0xFF) << 24);
    publicKeyBytes[2] = ((x32[5] >> 8) & 0xFFFFFF) | ((x32[6] & 0xFF) << 24);
    publicKeyBytes[3] = ((x32[4] >> 8) & 0xFFFFFF) | ((x32[5] & 0xFF) << 24);
    publicKeyBytes[4] = ((x32[3] >> 8) & 0xFFFFFF) | ((x32[4] & 0xFF) << 24);
    publicKeyBytes[5] = ((x32[2] >> 8) & 0xFFFFFF) | ((x32[3] & 0xFF) << 24);
    publicKeyBytes[6] = ((x32[1] >> 8) & 0xFFFFFF) | ((x32[2] & 0xFF) << 24);
    publicKeyBytes[7] = ((x32[0] >> 8) & 0xFFFFFF) | ((x32[1] & 0xFF) << 24);
    // Emulate CUDA's __byte_perm(x32[0], 0x80, 0x0456) = {input[6], input[5], input[4], input[0]} = {0, 0, 0x80, x32[0][0]}
    publicKeyBytes[8] = 0x00800000 | ((x32[0] & 0xFF) << 24);
    publicKeyBytes[9] = 0;
    publicKeyBytes[10] = 0;
    publicKeyBytes[11] = 0;
    publicKeyBytes[12] = 0;
    publicKeyBytes[13] = 0;
    publicKeyBytes[14] = 0;
    publicKeyBytes[15] = 0x108;
    
    // Odd (03 prefix) - only first word differs (prefix is in high byte now)
    publicKeyBytes2[0] = (publicKeyBytes[0] & 0x00FFFFFF) | 0x03000000;
    publicKeyBytes2[1] = publicKeyBytes[1];
    publicKeyBytes2[2] = publicKeyBytes[2];
    publicKeyBytes2[3] = publicKeyBytes[3];
    publicKeyBytes2[4] = publicKeyBytes[4];
    publicKeyBytes2[5] = publicKeyBytes[5];
    publicKeyBytes2[6] = publicKeyBytes[6];
    publicKeyBytes2[7] = publicKeyBytes[7];
    publicKeyBytes2[8] = publicKeyBytes[8];
    publicKeyBytes2[9] = 0;
    publicKeyBytes2[10] = 0;
    publicKeyBytes2[11] = 0;
    publicKeyBytes2[12] = 0;
    publicKeyBytes2[13] = 0;
    publicKeyBytes2[14] = 0;
    publicKeyBytes2[15] = 0x108;
    
    // Hash even key
    SHA256Initialize(s);
    SHA256Transform(s, publicKeyBytes);
    
    #pragma unroll
    for (int i = 0; i < 8; i++)
        s[i] = bswap32(s[i]);
    
    s[8] = 0x80;
    s[9] = 0;
    s[10] = 0;
    s[11] = 0;
    s[12] = 0;
    s[13] = 0;
    s[14] = 0x100;
    s[15] = 0;
    
    RIPEMD160Initialize(h1);
    RIPEMD160Transform(h1, s);
    
    // Hash odd key
    SHA256Initialize(s);
    SHA256Transform(s, publicKeyBytes2);
    
    #pragma unroll
    for (int i = 0; i < 8; i++)
        s[i] = bswap32(s[i]);
    
    s[8] = 0x80;
    s[9] = 0;
    s[10] = 0;
    s[11] = 0;
    s[12] = 0;
    s[13] = 0;
    s[14] = 0x100;
    s[15] = 0;
    
    RIPEMD160Initialize(h2);
    RIPEMD160Transform(h2, s);
}
