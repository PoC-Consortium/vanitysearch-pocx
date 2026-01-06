//! secp256k1 field element arithmetic (mod p)
//! p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F

#![allow(clippy::needless_range_loop)] // Indexed loops clearer for low-level math

use std::ops::{Add, Mul, Neg, Sub};

/// Prime field element for secp256k1
/// p = 2^256 - 2^32 - 977
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FieldElement {
    pub d: [u64; 4],
}

// Field prime p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
const P: [u64; 4] = [
    0xFFFFFFFFFFFFFC2F,
    0xFFFFFFFFFFFFFFFE,
    0xFFFFFFFFFFFFFFFF,
    0xFFFFFFFFFFFFFFFF,
];

impl FieldElement {
    pub const ZERO: Self = Self { d: [0, 0, 0, 0] };
    pub const ONE: Self = Self { d: [1, 0, 0, 0] };

    #[inline]
    pub fn new(d: [u64; 4]) -> Self {
        Self { d }
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

    #[inline]
    pub fn is_odd(&self) -> bool {
        self.d[0] & 1 == 1
    }

    /// Modular addition
    pub fn add(&self, other: &Self) -> Self {
        let mut r = [0u64; 4];
        let mut carry = 0u64;

        for i in 0..4 {
            let (sum, c1) = self.d[i].overflowing_add(other.d[i]);
            let (sum, c2) = sum.overflowing_add(carry);
            r[i] = sum;
            carry = (c1 as u64) + (c2 as u64);
        }

        // Reduce mod p if needed
        let mut result = Self { d: r };
        if carry != 0 || result.gte_p() {
            result.sub_p();
        }
        result
    }

    /// Modular subtraction
    pub fn sub(&self, other: &Self) -> Self {
        let mut r = [0u64; 4];
        let mut borrow = 0u64;

        for i in 0..4 {
            let (diff, b1) = self.d[i].overflowing_sub(other.d[i]);
            let (diff, b2) = diff.overflowing_sub(borrow);
            r[i] = diff;
            borrow = (b1 as u64) + (b2 as u64);
        }

        let mut result = Self { d: r };
        if borrow != 0 {
            result.add_p();
        }
        result
    }

    /// Modular multiplication using Montgomery reduction optimized for secp256k1
    pub fn mul(&self, other: &Self) -> Self {
        self.mul_ref(other)
    }

    /// Modular multiplication (reference version)
    pub fn mul_ref(&self, other: &Self) -> Self {
        // Full 512-bit multiplication
        let mut t = [0u64; 8];

        for i in 0..4 {
            let mut carry = 0u128;
            for j in 0..4 {
                let prod = (self.d[i] as u128) * (other.d[j] as u128) + (t[i + j] as u128) + carry;
                t[i + j] = prod as u64;
                carry = prod >> 64;
            }
            t[i + 4] = carry as u64;
        }

        // Reduce mod p using secp256k1 specific reduction
        // p = 2^256 - 0x1000003D1
        self.reduce_512(&t)
    }

    /// Reduce 512-bit number mod p
    fn reduce_512(&self, t: &[u64; 8]) -> Self {
        // t[0..4] + t[4..8] * 2^256 mod p
        // = t[0..4] + t[4..8] * 0x1000003D1 mod p

        let mut r = [0u64; 5];
        let k: u64 = 0x1000003D1;

        // r = t[0..4]
        r[0] = t[0];
        r[1] = t[1];
        r[2] = t[2];
        r[3] = t[3];
        r[4] = 0;

        // r += t[4..8] * k
        let mut carry = 0u128;
        for i in 0..4 {
            let prod = (t[i + 4] as u128) * (k as u128) + (r[i] as u128) + carry;
            r[i] = prod as u64;
            carry = prod >> 64;
        }
        r[4] = carry as u64;

        // Final reduction if r >= p
        // r[4] contains overflow, reduce again
        if r[4] != 0 {
            let overflow = r[4];
            r[4] = 0;

            carry = (overflow as u128) * (k as u128) + (r[0] as u128);
            r[0] = carry as u64;
            carry >>= 64;

            for i in 1..4 {
                carry += r[i] as u128;
                r[i] = carry as u64;
                carry >>= 64;
            }
        }

        let mut result = Self {
            d: [r[0], r[1], r[2], r[3]],
        };
        if result.gte_p() {
            result.sub_p();
        }
        result
    }

    /// Modular square
    #[inline]
    pub fn sqr(&self) -> Self {
        self.mul(self)
    }

    /// Modular negation
    pub fn neg(&self) -> Self {
        if self.is_zero() {
            *self
        } else {
            let mut r = [0u64; 4];
            let mut borrow = 0u64;

            for i in 0..4 {
                let (diff, b1) = P[i].overflowing_sub(self.d[i]);
                let (diff, b2) = diff.overflowing_sub(borrow);
                r[i] = diff;
                borrow = (b1 as u64) + (b2 as u64);
            }

            Self { d: r }
        }
    }

    /// Modular inverse using Fermat's little theorem
    /// a^(-1) = a^(p-2) mod p
    pub fn inv(&self) -> Self {
        // Use addition chain optimized for secp256k1
        // p - 2 = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2D

        let x2 = self.sqr().mul_ref(self);
        let x3 = x2.sqr().mul_ref(self);
        let x6 = x3.sqr().sqr().sqr().mul_ref(&x3);
        let x9 = x6.sqr().sqr().sqr().mul_ref(&x3);
        let x11 = x9.sqr().sqr().mul_ref(&x2);
        let x22 = (0..11).fold(x11, |acc, _| acc.sqr()).mul_ref(&x11);
        let x44 = (0..22).fold(x22, |acc, _| acc.sqr()).mul_ref(&x22);
        let x88 = (0..44).fold(x44, |acc, _| acc.sqr()).mul_ref(&x44);
        let x176 = (0..88).fold(x88, |acc, _| acc.sqr()).mul_ref(&x88);
        let x220 = (0..44).fold(x176, |acc, _| acc.sqr()).mul_ref(&x44);
        let x223 = x220.sqr().sqr().sqr().mul_ref(&x3);

        // t1 = x223 << 23
        let t1 = (0..23).fold(x223, |acc, _| acc.sqr());
        // t1 = t1 * x22
        let t1 = t1.mul_ref(&x22);
        // t1 = t1 << 5
        let t1 = (0..5).fold(t1, |acc, _| acc.sqr());
        // t1 = t1 * x
        let t1 = t1.mul_ref(self);
        // t1 = t1 << 3
        let t1 = (0..3).fold(t1, |acc, _| acc.sqr());
        // t1 = t1 * x2
        let t1 = t1.mul_ref(&x2);
        // t1 = t1 << 2
        let t1 = t1.sqr().sqr();
        // t1 = t1 * x
        t1.mul_ref(self)
    }

    /// Get square root if exists (returns None if not a quadratic residue)
    pub fn sqrt(&self) -> Option<Self> {
        // For secp256k1, p ≡ 3 (mod 4), so sqrt(a) = a^((p+1)/4) if exists
        // (p + 1) / 4 = 0x3FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFBFFFFF0C

        let x2 = self.sqr().mul_ref(self);
        let x3 = x2.sqr().mul_ref(self);
        let x6 = x3.sqr().sqr().sqr().mul_ref(&x3);
        let x9 = x6.sqr().sqr().sqr().mul_ref(&x3);
        let x11 = x9.sqr().sqr().mul_ref(&x2);
        let x22 = (0..11).fold(x11, |acc, _| acc.sqr()).mul_ref(&x11);
        let x44 = (0..22).fold(x22, |acc, _| acc.sqr()).mul_ref(&x22);
        let x88 = (0..44).fold(x44, |acc, _| acc.sqr()).mul_ref(&x44);
        let x176 = (0..88).fold(x88, |acc, _| acc.sqr()).mul_ref(&x88);
        let x220 = (0..44).fold(x176, |acc, _| acc.sqr()).mul_ref(&x44);
        let x223 = x220.sqr().sqr().sqr().mul_ref(&x3);

        let t1 = (0..23).fold(x223, |acc, _| acc.sqr());
        let t1 = t1.mul_ref(&x22);
        let t1 = (0..6).fold(t1, |acc, _| acc.sqr());
        let t1 = t1.mul_ref(&x2);
        let t1 = t1.sqr().sqr();
        let result = t1.mul_ref(self);

        // Verify
        if result.sqr() == *self {
            Some(result)
        } else {
            None
        }
    }

    #[inline]
    fn gte_p(&self) -> bool {
        for i in (0..4).rev() {
            if self.d[i] > P[i] {
                return true;
            }
            if self.d[i] < P[i] {
                return false;
            }
        }
        true // equal
    }

    #[inline]
    fn sub_p(&mut self) {
        let mut borrow = 0u64;
        for i in 0..4 {
            let (diff, b1) = self.d[i].overflowing_sub(P[i]);
            let (diff, b2) = diff.overflowing_sub(borrow);
            self.d[i] = diff;
            borrow = (b1 as u64) + (b2 as u64);
        }
    }

    #[inline]
    fn add_p(&mut self) {
        let mut carry = 0u64;
        for i in 0..4 {
            let (sum, c1) = self.d[i].overflowing_add(P[i]);
            let (sum, c2) = sum.overflowing_add(carry);
            self.d[i] = sum;
            carry = (c1 as u64) + (c2 as u64);
        }
    }
}

impl Add for FieldElement {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        FieldElement::add(&self, &other)
    }
}

impl Sub for FieldElement {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        FieldElement::sub(&self, &other)
    }
}

impl Mul for FieldElement {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        FieldElement::mul(&self, &other)
    }
}

impl Neg for FieldElement {
    type Output = Self;
    fn neg(self) -> Self {
        FieldElement::neg(&self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_field_add() {
        let a = FieldElement::new([1, 0, 0, 0]);
        let b = FieldElement::new([2, 0, 0, 0]);
        let c = a.add(b);
        assert_eq!(c.d[0], 3);
    }

    #[test]
    fn test_field_mul() {
        let a = FieldElement::new([2, 0, 0, 0]);
        let b = FieldElement::new([3, 0, 0, 0]);
        let c = a.mul(b);
        assert_eq!(c.d[0], 6);
    }

    #[test]
    fn test_field_inv() {
        let a = FieldElement::new([7, 0, 0, 0]);
        let inv = a.inv();
        let prod = a.mul(inv);
        assert_eq!(prod, FieldElement::ONE);
    }
}
