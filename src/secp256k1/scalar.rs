//! secp256k1 scalar arithmetic (mod n)
//! n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141

#![allow(clippy::needless_range_loop)] // Indexed loops clearer for low-level math

use std::ops::{Add, Sub, Mul, Neg};

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
                bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3],
                bytes[offset + 4], bytes[offset + 5], bytes[offset + 6], bytes[offset + 7],
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
            let high = Scalar { d: [t[4], t[5], t[6], t[7]] };
            
            // 2^256 mod n = 0x14551231950B75FC4402DA1732FC9BEBF
            let k = Scalar::new([
                0x4402DA1732FC9BEB,
                0x0000000000000001,
                0x0000000000000000,
                0x0000000000000000,
            ]);
            
            // Actually we need to handle this properly
            // For now, do iterative reduction
            let mut result = Scalar { d: [r[0], r[1], r[2], r[3]] };
            
            // Add contribution from high part
            let contrib = Scalar::mul_ref(&high, &k);
            result = Scalar::add_ref(&result, &contrib);
            
            return result;
        }

        let mut result = Scalar { d: [r[0], r[1], r[2], r[3]] };
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
}
