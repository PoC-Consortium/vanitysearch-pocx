//! secp256k1 scalar arithmetic (mod n)
//! n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141

#![allow(clippy::needless_range_loop)] // Indexed loops clearer for low-level math

use std::ops::{Add, Mul, Neg, Sub};

/// Scalar element for secp256k1 (private key domain)
/// n = order of the curve
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Scalar {
    pub d: [u64; 4],
}

// Curve order n
const N: [u64; 4] = [
    0xBFD25E8CD0364141,
    0xBAAEDCE6AF48A03B,
    0xFFFFFFFFFFFFFFFE,
    0xFFFFFFFFFFFFFFFF,
];

// n/2 for checking if scalar is "high"
const N_HALF: [u64; 4] = [
    0xDFE92F46681B20A0,
    0x5D576E7357A4501D,
    0xFFFFFFFFFFFFFFFF,
    0x7FFFFFFFFFFFFFFF,
];

impl Scalar {
    pub const ZERO: Self = Self { d: [0, 0, 0, 0] };
    pub const ONE: Self = Self { d: [1, 0, 0, 0] };

    #[inline]
    pub fn new(d: [u64; 4]) -> Self {
        Self { d }
    }

    #[inline]
    pub fn from_u64(val: u64) -> Self {
        Self { d: [val, 0, 0, 0] }
    }

    #[inline]
    pub fn from_bytes(bytes: &[u8; 32]) -> Self {
        let mut d = [0u64; 4];
        for i in 0..4 {
            let offset = (3 - i) * 8;
            d[i] = u64::from_be_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
                bytes[offset + 4],
                bytes[offset + 5],
                bytes[offset + 6],
                bytes[offset + 7],
            ]);
        }
        Self { d }
    }

    #[inline]
    pub fn to_bytes(&self) -> [u8; 32] {
        let mut bytes = [0u8; 32];
        for i in 0..4 {
            let offset = (3 - i) * 8;
            let b = self.d[i].to_be_bytes();
            bytes[offset..offset + 8].copy_from_slice(&b);
        }
        bytes
    }

    #[inline]
    pub fn is_zero(&self) -> bool {
        self.d[0] == 0 && self.d[1] == 0 && self.d[2] == 0 && self.d[3] == 0
    }

    /// Check if scalar is valid (0 < s < n)
    pub fn is_valid(&self) -> bool {
        !self.is_zero() && !self.gte_n()
    }

    /// Check if scalar is "high" (s > n/2)
    pub fn is_high(&self) -> bool {
        for i in (0..4).rev() {
            if self.d[i] > N_HALF[i] {
                return true;
            }
            if self.d[i] < N_HALF[i] {
                return false;
            }
        }
        true
    }

    /// Modular addition
    pub fn add(&self, other: &Self) -> Self {
        Self::add_ref(self, other)
    }

    /// Modular addition (reference version)
    pub fn add_ref(a: &Self, b: &Self) -> Self {
        let mut r = [0u64; 4];
        let mut carry = 0u64;

        for i in 0..4 {
            let (sum, c1) = a.d[i].overflowing_add(b.d[i]);
            let (sum, c2) = sum.overflowing_add(carry);
            r[i] = sum;
            carry = (c1 as u64) + (c2 as u64);
        }

        let mut result = Self { d: r };
        if carry != 0 || result.gte_n() {
            result.sub_n();
        }
        result
    }

    /// Modular subtraction
    pub fn sub(&self, other: &Self) -> Self {
        Self::sub_ref(self, other)
    }

    /// Modular subtraction (reference version)
    pub fn sub_ref(a: &Self, b: &Self) -> Self {
        let mut r = [0u64; 4];
        let mut borrow = 0u64;

        for i in 0..4 {
            let (diff, b1) = a.d[i].overflowing_sub(b.d[i]);
            let (diff, b2) = diff.overflowing_sub(borrow);
            r[i] = diff;
            borrow = (b1 as u64) + (b2 as u64);
        }

        let mut result = Self { d: r };
        if borrow != 0 {
            result.add_n();
        }
        result
    }

    /// Modular negation
    pub fn neg(&self) -> Self {
        if self.is_zero() {
            *self
        } else {
            let mut r = [0u64; 4];
            let mut borrow = 0u64;

            for i in 0..4 {
                let (diff, b1) = N[i].overflowing_sub(self.d[i]);
                let (diff, b2) = diff.overflowing_sub(borrow);
                r[i] = diff;
                borrow = (b1 as u64) + (b2 as u64);
            }

            Self { d: r }
        }
    }

    /// Modular multiplication
    pub fn mul(&self, other: &Self) -> Self {
        Self::mul_ref(self, other)
    }

    /// Modular multiplication (reference version)
    pub fn mul_ref(a: &Self, b: &Self) -> Self {
        let mut t = [0u64; 8];

        for i in 0..4 {
            let mut carry = 0u128;
            for j in 0..4 {
                let prod = (a.d[i] as u128) * (b.d[j] as u128) + (t[i + j] as u128) + carry;
                t[i + j] = prod as u64;
                carry = prod >> 64;
            }
            t[i + 4] = carry as u64;
        }

        Self::reduce_512_static(&t)
    }

    /// Reduce 512-bit number mod n
    fn reduce_512_static(t: &[u64; 8]) -> Self {
        // Barrett reduction for secp256k1 order
        // This is a simplified version - proper Barrett would be more complex

        let mut r = [0u64; 5];

        // Copy lower 256 bits
        r[0] = t[0];
        r[1] = t[1];
        r[2] = t[2];
        r[3] = t[3];
        r[4] = 0;

        // Reduce high bits
        // For simplicity, we do multiple subtractions if needed
        // In production, use proper Barrett reduction

        if t[4] != 0 || t[5] != 0 || t[6] != 0 || t[7] != 0 {
            // Compute high * something and subtract
            // This is a simplified reduction
            let high = Scalar {
                d: [t[4], t[5], t[6], t[7]],
            };

            // 2^256 mod n = 0x14551231950B75FC4402DA1732FC9BEBF
            // In little-endian 64-bit limbs:
            let k = Scalar::new([
                0x402DA1732FC9BEBF,
                0x4551231950B75FC4,
                0x0000000000000001,
                0x0000000000000000,
            ]);

            // Actually we need to handle this properly
            // For now, do iterative reduction
            let mut result = Scalar {
                d: [r[0], r[1], r[2], r[3]],
            };

            // Add contribution from high part
            let contrib = Scalar::mul_ref(&high, &k);
            result = Scalar::add_ref(&result, &contrib);

            return result;
        }

        let mut result = Scalar {
            d: [r[0], r[1], r[2], r[3]],
        };
        while result.gte_n() {
            result.sub_n();
        }
        result
    }

    /// Add scalar value (for key derivation)
    pub fn add_assign(&mut self, val: i64) {
        if val >= 0 {
            let other = Scalar::new([val as u64, 0, 0, 0]);
            *self = Scalar::add_ref(self, &other);
        } else {
            let other = Scalar::new([(-val) as u64, 0, 0, 0]);
            *self = Scalar::sub_ref(self, &other);
        }
    }

    /// Multiply by small constant
    pub fn mul_small(&self, k: u64) -> Self {
        let other = Scalar::new([k, 0, 0, 0]);
        Scalar::mul_ref(self, &other)
    }

    #[inline]
    fn gte_n(&self) -> bool {
        for i in (0..4).rev() {
            if self.d[i] > N[i] {
                return true;
            }
            if self.d[i] < N[i] {
                return false;
            }
        }
        true
    }

    #[inline]
    fn sub_n(&mut self) {
        let mut borrow = 0u64;
        for i in 0..4 {
            let (diff, b1) = self.d[i].overflowing_sub(N[i]);
            let (diff, b2) = diff.overflowing_sub(borrow);
            self.d[i] = diff;
            borrow = (b1 as u64) + (b2 as u64);
        }
    }

    #[inline]
    fn add_n(&mut self) {
        let mut carry = 0u64;
        for i in 0..4 {
            let (sum, c1) = self.d[i].overflowing_add(N[i]);
            let (sum, c2) = sum.overflowing_add(carry);
            self.d[i] = sum;
            carry = (c1 as u64) + (c2 as u64);
        }
    }
}

impl Add for Scalar {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Scalar::add(&self, &other)
    }
}

impl Sub for Scalar {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        Scalar::sub(&self, &other)
    }
}

impl Mul for Scalar {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        Scalar::mul(&self, &other)
    }
}

impl Neg for Scalar {
    type Output = Self;
    fn neg(self) -> Self {
        Scalar::neg(&self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_add() {
        let a = Scalar::new([1, 0, 0, 0]);
        let b = Scalar::new([2, 0, 0, 0]);
        let c = a.add(b);
        assert_eq!(c.d[0], 3);
    }

    #[test]
    fn test_scalar_neg() {
        let a = Scalar::new([1, 0, 0, 0]);
        let neg_a = a.neg();
        let sum = a.add(neg_a);
        assert!(sum.is_zero());
    }

    #[test]
    fn test_scalar_mul_lambda() {
        // Test: base_key = 0x31aaf575b15857236b77f094a13a1dd8749b2aa0c33e4bb0b2c90fb30e8a2100
        // base_key + 5 = 0x31aaf575b15857236b77f094a13a1dd8749b2aa0c33e4bb0b2c90fb30e8a2105
        // (base_key + 5) * lambda1 = 0xbb8cd229553563a7990c68ba4e4a39c40a47c7898f6c289668bc0493f768072b
        
        let base_key_bytes: [u8; 32] = [
            0x31, 0xaa, 0xf5, 0x75, 0xb1, 0x58, 0x57, 0x23,
            0x6b, 0x77, 0xf0, 0x94, 0xa1, 0x3a, 0x1d, 0xd8,
            0x74, 0x9b, 0x2a, 0xa0, 0xc3, 0x3e, 0x4b, 0xb0,
            0xb2, 0xc9, 0x0f, 0xb3, 0x0e, 0x8a, 0x21, 0x00,
        ];
        let base_key = Scalar::from_bytes(&base_key_bytes);
        
        let mut offset_bytes = [0u8; 32];
        offset_bytes[31] = 5;
        let key = Scalar::add_ref(&base_key, &Scalar::from_bytes(&offset_bytes));
        
        // Verify addition
        let expected_sum: [u8; 32] = [
            0x31, 0xaa, 0xf5, 0x75, 0xb1, 0x58, 0x57, 0x23,
            0x6b, 0x77, 0xf0, 0x94, 0xa1, 0x3a, 0x1d, 0xd8,
            0x74, 0x9b, 0x2a, 0xa0, 0xc3, 0x3e, 0x4b, 0xb0,
            0xb2, 0xc9, 0x0f, 0xb3, 0x0e, 0x8a, 0x21, 0x05,
        ];
        assert_eq!(key.to_bytes(), expected_sum, "Addition failed");
        
        let lambda1_bytes: [u8; 32] = [
            0x53, 0x63, 0xad, 0x4c, 0xc0, 0x5c, 0x30, 0xe0,
            0xa5, 0x26, 0x1c, 0x02, 0x88, 0x12, 0x64, 0x5a,
            0x12, 0x2e, 0x22, 0xea, 0x20, 0x81, 0x66, 0x78,
            0xdf, 0x02, 0x96, 0x7c, 0x1b, 0x23, 0xbd, 0x72,
        ];
        let lambda1 = Scalar::from_bytes(&lambda1_bytes);
        let result = Scalar::mul_ref(&key, &lambda1);
        
        let expected_result: [u8; 32] = [
            0xbb, 0x8c, 0xd2, 0x29, 0x55, 0x35, 0x63, 0xa7,
            0x99, 0x0c, 0x68, 0xba, 0x4e, 0x4a, 0x39, 0xc4,
            0x0a, 0x47, 0xc7, 0x89, 0x8f, 0x6c, 0x28, 0x96,
            0x68, 0xbc, 0x04, 0x93, 0xf7, 0x68, 0x07, 0x2b,
        ];
        assert_eq!(result.to_bytes(), expected_result, "Multiplication failed");
    }
}
