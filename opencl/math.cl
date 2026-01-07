/*
 * VanitySearch-POCX OpenCL Math Library
 * secp256k1 field arithmetic for OpenCL
 * 
 * Ported from VanitySearch GPUMath.h (CUDA) by Jean Luc PONS
 */

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

// ---------------------------------------------------------------------------------------
// 256-bit integer representation using 4 x 64-bit limbs
// We need 5 blocks for ModInv intermediate computations
// ---------------------------------------------------------------------------------------

#define NBBLOCK 5

// secp256k1 prime: P = 2^256 - 2^32 - 977
// P = 0xFFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE FFFFFC2F
#define P0 0xFFFFFFFEFFFFFC2FUL
#define P1 0xFFFFFFFFFFFFFFFFUL
#define P2 0xFFFFFFFFFFFFFFFFUL
#define P3 0xFFFFFFFFFFFFFFFFUL

// 64-bit LSB of negative inverse of P (mod 2^64)
#define MM64 0xD838091DD2253531UL

// Mask for 62-bit operations
#define MSK62 0x3FFFFFFFFFFFFFFFUL

// secp256k1 endomorphism constants
__constant ulong _beta[4] = {
    0xC1396C28719501EEUL, 0x9CF0497512F58995UL,
    0x6E64479EAC3434E9UL, 0x7AE96A2B657C0710UL
};

__constant ulong _beta2[4] = {
    0x3EC693D68E6AFA40UL, 0x630FB68AED0A766AUL,
    0x919BB86153CBCB16UL, 0x851695D49A83F8EFUL
};

// ---------------------------------------------------------------------------------------
// Helper macros for carry-chain arithmetic
// OpenCL doesn't have inline PTX, so we emulate with standard operations
// ---------------------------------------------------------------------------------------

// Add with carry out
inline ulong add_cc(ulong a, ulong b, ulong *carry) {
    ulong r = a + b;
    *carry = (r < a) ? 1UL : 0UL;
    return r;
}

// Add with carry in and carry out
inline ulong addc_cc(ulong a, ulong b, ulong carry_in, ulong *carry_out) {
    ulong r = a + b + carry_in;
    *carry_out = (r < a || (carry_in && r == a)) ? 1UL : 0UL;
    return r;
}

// Sub with borrow out
inline ulong sub_cc(ulong a, ulong b, ulong *borrow) {
    ulong r = a - b;
    *borrow = (a < b) ? 1UL : 0UL;
    return r;
}

// Sub with borrow in and borrow out
inline ulong subc_cc(ulong a, ulong b, ulong borrow_in, ulong *borrow_out) {
    ulong t = a - borrow_in;
    ulong r = t - b;
    *borrow_out = (a < borrow_in || t < b) ? 1UL : 0UL;
    return r;
}

// 64x64 -> 128 multiplication
inline ulong2 mul64(ulong a, ulong b) {
    // Split into 32-bit parts
    ulong a_lo = a & 0xFFFFFFFFUL;
    ulong a_hi = a >> 32;
    ulong b_lo = b & 0xFFFFFFFFUL;
    ulong b_hi = b >> 32;
    
    // Partial products
    ulong p0 = a_lo * b_lo;
    ulong p1 = a_lo * b_hi;
    ulong p2 = a_hi * b_lo;
    ulong p3 = a_hi * b_hi;
    
    // Combine
    ulong carry = 0;
    ulong mid = p1 + p2;
    if (mid < p1) carry = 1UL << 32;
    
    ulong lo = p0 + (mid << 32);
    ulong hi = p3 + (mid >> 32) + carry + (lo < p0 ? 1UL : 0UL);
    
    return (ulong2)(lo, hi);
}

// Note: mad_hi is a built-in OpenCL function, but we need this custom version
// for proper behavior with our 64-bit multiply-add operation
inline ulong mad_hi_custom(ulong a, ulong b, ulong c) {
    ulong2 p = mul64(a, b);
    ulong lo = p.x + c;
    return p.y + (lo < p.x ? 1UL : 0UL);
}

// ---------------------------------------------------------------------------------------
// Basic predicates
// ---------------------------------------------------------------------------------------

inline bool IsPositive5(ulong *x) {
    return ((long)x[4]) >= 0L;
}

inline bool IsNegative5(ulong *x) {
    return ((long)x[4]) < 0L;
}

inline bool IsZero5(ulong *x) {
    return (x[0] | x[1] | x[2] | x[3] | x[4]) == 0UL;
}

inline bool IsOne5(ulong *x) {
    return x[0] == 1UL && x[1] == 0UL && x[2] == 0UL && x[3] == 0UL && x[4] == 0UL;
}

// ---------------------------------------------------------------------------------------
// Load/Store operations
// ---------------------------------------------------------------------------------------

inline void Load5(ulong *r, __private ulong *a) {
    r[0] = a[0]; r[1] = a[1]; r[2] = a[2]; r[3] = a[3]; r[4] = a[4];
}

inline void Load256(__private ulong *r, __private ulong *a) {
    r[0] = a[0]; r[1] = a[1]; r[2] = a[2]; r[3] = a[3];
}

inline void Load256G(__global ulong *keys, int idx, int stride, __private ulong *r) {
    r[0] = keys[idx];
    r[1] = keys[idx + stride];
    r[2] = keys[idx + 2 * stride];
    r[3] = keys[idx + 3 * stride];
}

inline void Store256G(__global ulong *keys, int idx, int stride, __private ulong *r) {
    keys[idx] = r[0];
    keys[idx + stride] = r[1];
    keys[idx + 2 * stride] = r[2];
    keys[idx + 3 * stride] = r[3];
}

// ---------------------------------------------------------------------------------------
// 256-bit addition: r = a + b (mod P)
// ---------------------------------------------------------------------------------------

inline void ModAdd256(__private ulong *r, __private ulong *a, __private ulong *b) {
    ulong c;
    r[0] = add_cc(a[0], b[0], &c);
    r[1] = addc_cc(a[1], b[1], c, &c);
    r[2] = addc_cc(a[2], b[2], c, &c);
    r[3] = addc_cc(a[3], b[3], c, &c);
    
    // Conditional subtract P if overflow
    ulong t = c;  // carry
    ulong T[4];
    T[0] = P0 & (-t);
    T[1] = P1 & (-t);
    T[2] = P2 & (-t);
    T[3] = P3 & (-t);
    
    ulong br;
    r[0] = sub_cc(r[0], T[0], &br);
    r[1] = subc_cc(r[1], T[1], br, &br);
    r[2] = subc_cc(r[2], T[2], br, &br);
    r[3] = subc_cc(r[3], T[3], br, &br);
}

// ---------------------------------------------------------------------------------------
// 256-bit subtraction: r = a - b (mod P)
// ---------------------------------------------------------------------------------------

inline void ModSub256(__private ulong *r, __private ulong *a, __private ulong *b) {
    ulong br;
    r[0] = sub_cc(a[0], b[0], &br);
    r[1] = subc_cc(a[1], b[1], br, &br);
    r[2] = subc_cc(a[2], b[2], br, &br);
    r[3] = subc_cc(a[3], b[3], br, &br);
    
    // Conditional add P if borrow
    ulong t = br;
    ulong T[4];
    T[0] = P0 & (-t);
    T[1] = P1 & (-t);
    T[2] = P2 & (-t);
    T[3] = P3 & (-t);
    
    ulong c;
    r[0] = add_cc(r[0], T[0], &c);
    r[1] = addc_cc(r[1], T[1], c, &c);
    r[2] = addc_cc(r[2], T[2], c, &c);
    r[3] = addc_cc(r[3], T[3], c, &c);
}

// In-place subtraction: r = r - b (mod P)
inline void ModSub256_ip(__private ulong *r, __private ulong *b) {
    ulong br;
    ulong t0 = sub_cc(r[0], b[0], &br);
    ulong t1 = subc_cc(r[1], b[1], br, &br);
    ulong t2 = subc_cc(r[2], b[2], br, &br);
    ulong t3 = subc_cc(r[3], b[3], br, &br);
    
    ulong t = br;
    ulong T[4];
    T[0] = P0 & (-t);
    T[1] = P1 & (-t);
    T[2] = P2 & (-t);
    T[3] = P3 & (-t);
    
    ulong c;
    r[0] = add_cc(t0, T[0], &c);
    r[1] = addc_cc(t1, T[1], c, &c);
    r[2] = addc_cc(t2, T[2], c, &c);
    r[3] = addc_cc(t3, T[3], c, &c);
}

// ---------------------------------------------------------------------------------------
// 256-bit negation: r = -a (mod P)
// ---------------------------------------------------------------------------------------

inline void ModNeg256(__private ulong *r, __private ulong *a) {
    ulong br;
    ulong t0 = sub_cc(0UL, a[0], &br);
    ulong t1 = subc_cc(0UL, a[1], br, &br);
    ulong t2 = subc_cc(0UL, a[2], br, &br);
    ulong t3 = subc_cc(0UL, a[3], br, &br);
    
    ulong c;
    r[0] = add_cc(t0, P0, &c);
    r[1] = addc_cc(t1, P1, c, &c);
    r[2] = addc_cc(t2, P2, c, &c);
    r[3] = addc_cc(t3, P3, c, &c);
}

inline void ModNeg256_ip(__private ulong *r) {
    ulong br;
    ulong t0 = sub_cc(0UL, r[0], &br);
    ulong t1 = subc_cc(0UL, r[1], br, &br);
    ulong t2 = subc_cc(0UL, r[2], br, &br);
    ulong t3 = subc_cc(0UL, r[3], br, &br);
    
    ulong c;
    r[0] = add_cc(t0, P0, &c);
    r[1] = addc_cc(t1, P1, c, &c);
    r[2] = addc_cc(t2, P2, c, &c);
    r[3] = addc_cc(t3, P3, c, &c);
}

// ---------------------------------------------------------------------------------------
// Overloaded versions for __constant memory (for lookup tables)
// ---------------------------------------------------------------------------------------

// Subtract constant from private: r = a - b (mod P) where a is __constant
inline void ModSub256C(__private ulong *r, __constant ulong *a, __private ulong *b) {
    ulong br;
    r[0] = sub_cc(a[0], b[0], &br);
    r[1] = subc_cc(a[1], b[1], br, &br);
    r[2] = subc_cc(a[2], b[2], br, &br);
    r[3] = subc_cc(a[3], b[3], br, &br);
    
    ulong t = br;
    ulong T[4];
    T[0] = P0 & (-t);
    T[1] = P1 & (-t);
    T[2] = P2 & (-t);
    T[3] = P3 & (-t);
    
    ulong c;
    r[0] = add_cc(r[0], T[0], &c);
    r[1] = addc_cc(r[1], T[1], c, &c);
    r[2] = addc_cc(r[2], T[2], c, &c);
    r[3] = addc_cc(r[3], T[3], c, &c);
}

// Subtract private from constant: r = a - b (mod P) where b is __constant
inline void ModSub256PC(__private ulong *r, __private ulong *a, __constant ulong *b) {
    ulong br;
    r[0] = sub_cc(a[0], b[0], &br);
    r[1] = subc_cc(a[1], b[1], br, &br);
    r[2] = subc_cc(a[2], b[2], br, &br);
    r[3] = subc_cc(a[3], b[3], br, &br);
    
    ulong t = br;
    ulong T[4];
    T[0] = P0 & (-t);
    T[1] = P1 & (-t);
    T[2] = P2 & (-t);
    T[3] = P3 & (-t);
    
    ulong c;
    r[0] = add_cc(r[0], T[0], &c);
    r[1] = addc_cc(r[1], T[1], c, &c);
    r[2] = addc_cc(r[2], T[2], c, &c);
    r[3] = addc_cc(r[3], T[3], c, &c);
}

// In-place subtract constant: r = r - b (mod P) where b is __constant
inline void ModSub256C_ip(__private ulong *r, __constant ulong *b) {
    ulong br;
    ulong t0 = sub_cc(r[0], b[0], &br);
    ulong t1 = subc_cc(r[1], b[1], br, &br);
    ulong t2 = subc_cc(r[2], b[2], br, &br);
    ulong t3 = subc_cc(r[3], b[3], br, &br);
    
    ulong t = br;
    ulong T[4];
    T[0] = P0 & (-t);
    T[1] = P1 & (-t);
    T[2] = P2 & (-t);
    T[3] = P3 & (-t);
    
    ulong c;
    r[0] = add_cc(t0, T[0], &c);
    r[1] = addc_cc(t1, T[1], c, &c);
    r[2] = addc_cc(t2, T[2], c, &c);
    r[3] = addc_cc(t3, T[3], c, &c);
}

// Negate constant: r = -a (mod P) where a is __constant
inline void ModNeg256C(__private ulong *r, __constant ulong *a) {
    ulong br;
    ulong t0 = sub_cc(0UL, a[0], &br);
    ulong t1 = subc_cc(0UL, a[1], br, &br);
    ulong t2 = subc_cc(0UL, a[2], br, &br);
    ulong t3 = subc_cc(0UL, a[3], br, &br);
    
    ulong c;
    r[0] = add_cc(t0, P0, &c);
    r[1] = addc_cc(t1, P1, c, &c);
    r[2] = addc_cc(t2, P2, c, &c);
    r[3] = addc_cc(t3, P3, c, &c);
}

// Multiply with constant: r = a * b (mod P) where b is __constant
// Mirrors CUDA implementation exactly for correctness
void ModMultC(__private ulong *r, __private ulong *a, __constant ulong *b) {
    ulong r512[8];
    ulong t[5];
    ulong c;
    
    r512[5] = 0UL;
    r512[6] = 0UL;
    r512[7] = 0UL;
    
    // 256 x 256 multiplication -> 512 bits
    // Using CUDA's UMult pattern
    
    // r512 = a * b[0]
    ulong2 p;
    p = mul64(a[0], b[0]); r512[0] = p.x; t[0] = p.y;
    p = mul64(a[1], b[0]); r512[1] = p.x; t[1] = p.y;
    p = mul64(a[2], b[0]); r512[2] = p.x; t[2] = p.y;
    p = mul64(a[3], b[0]); r512[3] = p.x; t[3] = p.y;
    
    r512[1] = add_cc(r512[1], t[0], &c);
    r512[2] = addc_cc(r512[2], t[1], c, &c);
    r512[3] = addc_cc(r512[3], t[2], c, &c);
    r512[4] = addc_cc(0UL, t[3], c, &c);
    
    // t = a * b[1]
    p = mul64(a[0], b[1]); t[0] = p.x; ulong h0 = p.y;
    p = mul64(a[1], b[1]); t[1] = p.x; ulong h1 = p.y;
    p = mul64(a[2], b[1]); t[2] = p.x; ulong h2 = p.y;
    p = mul64(a[3], b[1]); t[3] = p.x; ulong h3 = p.y;
    
    t[1] = add_cc(t[1], h0, &c);
    t[2] = addc_cc(t[2], h1, c, &c);
    t[3] = addc_cc(t[3], h2, c, &c);
    t[4] = addc_cc(0UL, h3, c, &c);
    
    r512[1] = add_cc(r512[1], t[0], &c);
    r512[2] = addc_cc(r512[2], t[1], c, &c);
    r512[3] = addc_cc(r512[3], t[2], c, &c);
    r512[4] = addc_cc(r512[4], t[3], c, &c);
    r512[5] = addc_cc(r512[5], t[4], c, &c);
    
    // t = a * b[2]
    p = mul64(a[0], b[2]); t[0] = p.x; h0 = p.y;
    p = mul64(a[1], b[2]); t[1] = p.x; h1 = p.y;
    p = mul64(a[2], b[2]); t[2] = p.x; h2 = p.y;
    p = mul64(a[3], b[2]); t[3] = p.x; h3 = p.y;
    
    t[1] = add_cc(t[1], h0, &c);
    t[2] = addc_cc(t[2], h1, c, &c);
    t[3] = addc_cc(t[3], h2, c, &c);
    t[4] = addc_cc(0UL, h3, c, &c);
    
    r512[2] = add_cc(r512[2], t[0], &c);
    r512[3] = addc_cc(r512[3], t[1], c, &c);
    r512[4] = addc_cc(r512[4], t[2], c, &c);
    r512[5] = addc_cc(r512[5], t[3], c, &c);
    r512[6] = addc_cc(r512[6], t[4], c, &c);
    
    // t = a * b[3]
    p = mul64(a[0], b[3]); t[0] = p.x; h0 = p.y;
    p = mul64(a[1], b[3]); t[1] = p.x; h1 = p.y;
    p = mul64(a[2], b[3]); t[2] = p.x; h2 = p.y;
    p = mul64(a[3], b[3]); t[3] = p.x; h3 = p.y;
    
    t[1] = add_cc(t[1], h0, &c);
    t[2] = addc_cc(t[2], h1, c, &c);
    t[3] = addc_cc(t[3], h2, c, &c);
    t[4] = addc_cc(0UL, h3, c, &c);
    
    r512[3] = add_cc(r512[3], t[0], &c);
    r512[4] = addc_cc(r512[4], t[1], c, &c);
    r512[5] = addc_cc(r512[5], t[2], c, &c);
    r512[6] = addc_cc(r512[6], t[3], c, &c);
    r512[7] = addc_cc(r512[7], t[4], c, &c);
    
    // Reduce from 512 to 320 bits
    #define SECP256K1_C 0x1000003D1UL
    
    p = mul64(r512[4], SECP256K1_C); t[0] = p.x; h0 = p.y;
    p = mul64(r512[5], SECP256K1_C); t[1] = p.x; h1 = p.y;
    p = mul64(r512[6], SECP256K1_C); t[2] = p.x; h2 = p.y;
    p = mul64(r512[7], SECP256K1_C); t[3] = p.x; h3 = p.y;
    
    t[1] = add_cc(t[1], h0, &c);
    t[2] = addc_cc(t[2], h1, c, &c);
    t[3] = addc_cc(t[3], h2, c, &c);
    t[4] = addc_cc(0UL, h3, c, &c);
    
    r512[0] = add_cc(r512[0], t[0], &c);
    r512[1] = addc_cc(r512[1], t[1], c, &c);
    r512[2] = addc_cc(r512[2], t[2], c, &c);
    r512[3] = addc_cc(r512[3], t[3], c, &c);
    
    // Reduce from 320 to 256 bits
    ulong r4 = t[4] + c;
    
    p = mul64(r4, SECP256K1_C);
    r[0] = add_cc(r512[0], p.x, &c);
    r[1] = addc_cc(r512[1], p.y, c, &c);
    r[2] = addc_cc(r512[2], 0UL, c, &c);
    r[3] = addc_cc(r512[3], 0UL, c, &c);
}

// ---------------------------------------------------------------------------------------
// 256-bit multiplication: r = a * b (mod P)
// Mirrors CUDA implementation exactly for correctness
// ---------------------------------------------------------------------------------------

void ModMult(__private ulong *r, __private ulong *a, __private ulong *b) {
    ulong r512[8];
    ulong t[5];
    ulong c;
    
    r512[5] = 0UL;
    r512[6] = 0UL;
    r512[7] = 0UL;
    
    // 256 x 256 multiplication -> 512 bits
    // Using CUDA's UMult pattern: compute lo products, then add hi products with carry chain
    
    // r512 = a * b[0]
    ulong2 p;
    p = mul64(a[0], b[0]); r512[0] = p.x; t[0] = p.y;
    p = mul64(a[1], b[0]); r512[1] = p.x; t[1] = p.y;
    p = mul64(a[2], b[0]); r512[2] = p.x; t[2] = p.y;
    p = mul64(a[3], b[0]); r512[3] = p.x; t[3] = p.y;
    
    // Add high parts with carry chain
    r512[1] = add_cc(r512[1], t[0], &c);
    r512[2] = addc_cc(r512[2], t[1], c, &c);
    r512[3] = addc_cc(r512[3], t[2], c, &c);
    r512[4] = addc_cc(0UL, t[3], c, &c);
    
    // t = a * b[1]
    p = mul64(a[0], b[1]); t[0] = p.x; ulong h0 = p.y;
    p = mul64(a[1], b[1]); t[1] = p.x; ulong h1 = p.y;
    p = mul64(a[2], b[1]); t[2] = p.x; ulong h2 = p.y;
    p = mul64(a[3], b[1]); t[3] = p.x; ulong h3 = p.y;
    
    // t[1..4] += h[0..3] with carry
    t[1] = add_cc(t[1], h0, &c);
    t[2] = addc_cc(t[2], h1, c, &c);
    t[3] = addc_cc(t[3], h2, c, &c);
    t[4] = addc_cc(0UL, h3, c, &c);
    
    // r512 += t << 64
    r512[1] = add_cc(r512[1], t[0], &c);
    r512[2] = addc_cc(r512[2], t[1], c, &c);
    r512[3] = addc_cc(r512[3], t[2], c, &c);
    r512[4] = addc_cc(r512[4], t[3], c, &c);
    r512[5] = addc_cc(r512[5], t[4], c, &c);
    
    // t = a * b[2]
    p = mul64(a[0], b[2]); t[0] = p.x; h0 = p.y;
    p = mul64(a[1], b[2]); t[1] = p.x; h1 = p.y;
    p = mul64(a[2], b[2]); t[2] = p.x; h2 = p.y;
    p = mul64(a[3], b[2]); t[3] = p.x; h3 = p.y;
    
    t[1] = add_cc(t[1], h0, &c);
    t[2] = addc_cc(t[2], h1, c, &c);
    t[3] = addc_cc(t[3], h2, c, &c);
    t[4] = addc_cc(0UL, h3, c, &c);
    
    r512[2] = add_cc(r512[2], t[0], &c);
    r512[3] = addc_cc(r512[3], t[1], c, &c);
    r512[4] = addc_cc(r512[4], t[2], c, &c);
    r512[5] = addc_cc(r512[5], t[3], c, &c);
    r512[6] = addc_cc(r512[6], t[4], c, &c);
    
    // t = a * b[3]
    p = mul64(a[0], b[3]); t[0] = p.x; h0 = p.y;
    p = mul64(a[1], b[3]); t[1] = p.x; h1 = p.y;
    p = mul64(a[2], b[3]); t[2] = p.x; h2 = p.y;
    p = mul64(a[3], b[3]); t[3] = p.x; h3 = p.y;
    
    t[1] = add_cc(t[1], h0, &c);
    t[2] = addc_cc(t[2], h1, c, &c);
    t[3] = addc_cc(t[3], h2, c, &c);
    t[4] = addc_cc(0UL, h3, c, &c);
    
    r512[3] = add_cc(r512[3], t[0], &c);
    r512[4] = addc_cc(r512[4], t[1], c, &c);
    r512[5] = addc_cc(r512[5], t[2], c, &c);
    r512[6] = addc_cc(r512[6], t[3], c, &c);
    r512[7] = addc_cc(r512[7], t[4], c, &c);
    
    // Reduce from 512 to 320 bits
    // Mirrors CUDA: UMult(t, r512+4, 0x1000003D1ULL) then add
    #define SECP256K1_C 0x1000003D1UL
    
    // Compute t = r512[4..7] * SECP256K1_C (as 5 limbs)
    p = mul64(r512[4], SECP256K1_C); t[0] = p.x; h0 = p.y;
    p = mul64(r512[5], SECP256K1_C); t[1] = p.x; h1 = p.y;
    p = mul64(r512[6], SECP256K1_C); t[2] = p.x; h2 = p.y;
    p = mul64(r512[7], SECP256K1_C); t[3] = p.x; h3 = p.y;
    
    // Add high parts: t[1] += h0, t[2] += h1, t[3] += h2, t[4] = h3
    t[1] = add_cc(t[1], h0, &c);
    t[2] = addc_cc(t[2], h1, c, &c);
    t[3] = addc_cc(t[3], h2, c, &c);
    t[4] = addc_cc(0UL, h3, c, &c);
    
    // Add t to r512[0..3] with carry chain
    r512[0] = add_cc(r512[0], t[0], &c);
    r512[1] = addc_cc(r512[1], t[1], c, &c);
    r512[2] = addc_cc(r512[2], t[2], c, &c);
    r512[3] = addc_cc(r512[3], t[3], c, &c);
    
    // Reduce from 320 to 256 bits
    // CUDA: UADD1(t[4], 0ULL) captures carry into t[4]
    ulong r4 = t[4] + c;  // t[4] plus the carry from the add chain
    
    // r4 * SECP256K1_C
    p = mul64(r4, SECP256K1_C);
    r[0] = add_cc(r512[0], p.x, &c);
    r[1] = addc_cc(r512[1], p.y, c, &c);
    r[2] = addc_cc(r512[2], 0UL, c, &c);
    r[3] = addc_cc(r512[3], 0UL, c, &c);
}

// In-place multiply: r = a * r (mod P)
void ModMult_ip(__private ulong *r, __private ulong *a) {
    ulong b[4];
    Load256(b, r);
    ModMult(r, a, b);
}

// ---------------------------------------------------------------------------------------
// 256-bit squaring: r = a^2 (mod P)
// For correctness, just use ModMult. Can be optimized later.
// ---------------------------------------------------------------------------------------

void ModSqr(__private ulong *r, __private ulong *a) {
    ModMult(r, a, a);
}

// ---------------------------------------------------------------------------------------
// Modular inverse using extended binary GCD
// Based on Daniel J. Bernstein's safegcd algorithm
// ---------------------------------------------------------------------------------------

// 5-block negation
inline void Neg5(__private ulong *r) {
    ulong br;
    r[0] = sub_cc(0UL, r[0], &br);
    r[1] = subc_cc(0UL, r[1], br, &br);
    r[2] = subc_cc(0UL, r[2], br, &br);
    r[3] = subc_cc(0UL, r[3], br, &br);
    r[4] = subc_cc(0UL, r[4], br, &br);
}

// Add P to 5-block number
inline void AddP5(__private ulong *r) {
    ulong c;
    r[0] = add_cc(r[0], P0, &c);
    r[1] = addc_cc(r[1], P1, c, &c);
    r[2] = addc_cc(r[2], P2, c, &c);
    r[3] = addc_cc(r[3], P3, c, &c);
    r[4] = addc_cc(r[4], 0UL, c, &c);
}

// Subtract P from 5-block number
inline void SubP5(__private ulong *r) {
    ulong br;
    r[0] = sub_cc(r[0], P0, &br);
    r[1] = subc_cc(r[1], P1, br, &br);
    r[2] = subc_cc(r[2], P2, br, &br);
    r[3] = subc_cc(r[3], P3, br, &br);
    r[4] = subc_cc(r[4], 0UL, br, &br);
}

// Count trailing zeros
inline uint ctz64(ulong x) {
    if (x == 0) return 64;
    uint n = 0;
    if ((x & 0xFFFFFFFFUL) == 0) { n += 32; x >>= 32; }
    if ((x & 0xFFFFUL) == 0) { n += 16; x >>= 16; }
    if ((x & 0xFFUL) == 0) { n += 8; x >>= 8; }
    if ((x & 0xFUL) == 0) { n += 4; x >>= 4; }
    if ((x & 0x3UL) == 0) { n += 2; x >>= 2; }
    if ((x & 0x1UL) == 0) { n += 1; }
    return n;
}

// Count leading zeros  
inline uint clz64(ulong x) {
    if (x == 0) return 64;
    uint n = 0;
    if (x <= 0x00000000FFFFFFFFUL) { n += 32; x <<= 32; }
    if (x <= 0x0000FFFFFFFFFFFFUL) { n += 16; x <<= 16; }
    if (x <= 0x00FFFFFFFFFFFFFFUL) { n += 8; x <<= 8; }
    if (x <= 0x0FFFFFFFFFFFFFFFUL) { n += 4; x <<= 4; }
    if (x <= 0x3FFFFFFFFFFFFFFFUL) { n += 2; x <<= 2; }
    if (x <= 0x7FFFFFFFFFFFFFFFUL) { n += 1; }
    return n;
}

// Shift right by 62 bits
inline void ShiftR62(__private ulong *r) {
    r[0] = (r[1] << 2) | (r[0] >> 62);
    r[1] = (r[2] << 2) | (r[1] >> 62);
    r[2] = (r[3] << 2) | (r[2] >> 62);
    r[3] = (r[4] << 2) | (r[3] >> 62);
    r[4] = ((long)r[4]) >> 62;  // Sign-extending shift
}

// Modular inverse: R = R^(-1) mod P
// p-2 for Fermat's little theorem inverse
// p-2 = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2D
// In little-endian 64-bit words:
// w0 = 0xFFFFFFFEFFFFFC2D (bits 63-0)
// w1 = 0xFFFFFFFFFFFFFFFF (bits 127-64)
// w2 = 0xFFFFFFFFFFFFFFFF (bits 191-128)
// w3 = 0xFFFFFFFFFFFFFFFF (bits 255-192)
__constant ulong _p_minus_2[4] = {0xFFFFFFFEFFFFFC2DULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL};

// Modular inverse using Fermat's little theorem: a^(-1) = a^(p-2) mod p
// For secp256k1, p-2 = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2D
void ModInv(__private ulong *R) {
    ulong result[4], base[4];
    
    // base = R
    Load256(base, R);
    
    // Process MSB first (bit 255)
    // Since bit 255 of exp is 1, result = base
    Load256(result, base);
    
    // Process remaining bits 254 down to 0
    for (int i = 254; i >= 0; i--) {
        ModSqr(result, result);  // result = result^2
        
        int word = i / 64;
        int bit = i % 64;
        if (_p_minus_2[word] & (1ULL << bit)) {
            ModMult(result, result, base);  // result = result * base
        }
    }
    
    R[0] = result[0]; R[1] = result[1]; R[2] = result[2]; R[3] = result[3];
}

// ---------------------------------------------------------------------------------------
// Batch modular inverse for group
// Uses Montgomery's trick: inv(a) * inv(b) = inv(a*b)
// ---------------------------------------------------------------------------------------

#define HSIZE (GRP_SIZE / 2 - 1)

void ModInvGrouped(__private ulong r[][4], int count) {
    ulong subp[GRP_SIZE / 2 + 1][4];
    ulong newValue[4];
    ulong inverse[5];
    
    int tid = get_global_id(0);
    
    // Compute cumulative products
    Load256(subp[0], r[0]);
    for (int i = 1; i < count; i++) {
        ModMult(subp[i], subp[i - 1], r[i]);
    }
    
    // Invert the product
    Load256(inverse, subp[count - 1]);
    inverse[4] = 0;
    
    ModInv(inverse);
    
    // Propagate inverse back
    for (int i = count - 1; i > 0; i--) {
        ModMult(newValue, subp[i - 1], inverse);
        ModMult_ip(inverse, r[i]);
        Load256(r[i], newValue);
    }
    
    Load256(r[0], inverse);
}
