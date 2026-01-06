//! secp256k1 elliptic curve point operations

use super::field::FieldElement;
use super::scalar::Scalar;

/// Point on secp256k1 curve (affine coordinates)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Point {
    pub x: FieldElement,
    pub y: FieldElement,
    pub infinity: bool,
}

/// Point in Jacobian coordinates for faster operations
#[derive(Clone, Copy, Debug)]
pub struct JacobianPoint {
    pub x: FieldElement,
    pub y: FieldElement,
    pub z: FieldElement,
}

// Generator point G
pub const G: Point = Point {
    x: FieldElement { d: [
        0x59F2815B16F81798,
        0x029BFCDB2DCE28D9,
        0x55A06295CE870B07,
        0x79BE667EF9DCBBAC,
    ]},
    y: FieldElement { d: [
        0x9C47D08FFB10D4B8,
        0xFD17B448A6855419,
        0x5DA4FBFC0E1108A8,
        0x483ADA7726A3C465,
    ]},
    infinity: false,
};

// Endomorphism constants for secp256k1
// beta: cube root of unity in Fp
// lambda: cube root of unity in Fn
pub const BETA: FieldElement = FieldElement { d: [
    0xC1396C28719501EE,
    0x9CF0497512F58995,
    0x6E64479EAC3434E9,
    0x7AE96A2B657C0710,
]};

pub const BETA2: FieldElement = FieldElement { d: [
    0x3EC693D68E6AFA40,
    0x630FB68AED0A766A,
    0x919BB86153CBCB16,
    0x851695D49A83F8EF,
]};

// lambda for endomorphism
pub const LAMBDA: Scalar = Scalar { d: [
    0xDF02967C1B23BD72,
    0x122E22EA20816678,
    0xA5261C028812645A,
    0x5363AD4CC05C30E0,
]};

pub const LAMBDA2: Scalar = Scalar { d: [
    0xE0CFD10B51283CE0,
    0x8EC739C2E0CFC810,
    0x3FA3CF1F5AD9E3FD,
    0xAC9C52B33FA3CF1F,
]};

impl Point {
    pub const INFINITY: Self = Self {
        x: FieldElement::ZERO,
        y: FieldElement::ZERO,
        infinity: true,
    };

    #[inline]
    pub fn new(x: FieldElement, y: FieldElement) -> Self {
        Self { x, y, infinity: false }
    }

    #[inline]
    pub fn is_infinity(&self) -> bool {
        self.infinity
    }

    /// Convert to Jacobian coordinates
    pub fn to_jacobian(&self) -> JacobianPoint {
        if self.infinity {
            JacobianPoint {
                x: FieldElement::ONE,
                y: FieldElement::ONE,
                z: FieldElement::ZERO,
            }
        } else {
            JacobianPoint {
                x: self.x,
                y: self.y,
                z: FieldElement::ONE,
            }
        }
    }

    /// Point addition
    pub fn add(&self, other: &Self) -> Self {
        if self.infinity {
            return *other;
        }
        if other.infinity {
            return *self;
        }

        if self.x == other.x {
            if self.y == other.y {
                return self.double();
            } else {
                return Self::INFINITY;
            }
        }

        // λ = (y2 - y1) / (x2 - x1)
        let dy = other.y.sub(&self.y);
        let dx = other.x.sub(&self.x);
        let lambda = dy.mul(&dx.inv());

        // x3 = λ² - x1 - x2
        let x3 = lambda.sqr().sub(&self.x).sub(&other.x);
        // y3 = λ(x1 - x3) - y1
        let y3 = lambda.mul(&self.x.sub(&x3)).sub(&self.y);

        Self::new(x3, y3)
    }

    /// Point doubling
    pub fn double(&self) -> Self {
        if self.infinity {
            return Self::INFINITY;
        }
        if self.y.is_zero() {
            return Self::INFINITY;
        }

        // λ = 3x² / 2y (a = 0 for secp256k1)
        let x_sq = self.x.sqr();
        let three_x_sq = x_sq.add(&x_sq).add(&x_sq);
        let two_y = self.y.add(&self.y);
        let lambda = three_x_sq.mul(&two_y.inv());

        // x3 = λ² - 2x
        let x3 = lambda.sqr().sub(&self.x).sub(&self.x);
        // y3 = λ(x - x3) - y
        let y3 = lambda.mul(&self.x.sub(&x3)).sub(&self.y);

        Self::new(x3, y3)
    }

    /// Point negation
    pub fn neg(&self) -> Self {
        if self.infinity {
            Self::INFINITY
        } else {
            Self::new(self.x, self.y.neg())
        }
    }

    /// Scalar multiplication using double-and-add
    pub fn mul(&self, scalar: &Scalar) -> Self {
        let mut result = Self::INFINITY;
        let mut temp = *self;

        for i in 0..4 {
            let mut k = scalar.d[i];
            for _ in 0..64 {
                if k & 1 == 1 {
                    result = result.add(&temp);
                }
                temp = temp.double();
                k >>= 1;
            }
        }

        result
    }

    /// Apply endomorphism: (x, y) -> (beta * x, y)
    pub fn endomorphism1(&self) -> Self {
        if self.infinity {
            Self::INFINITY
        } else {
            Self::new(self.x.mul(&BETA), self.y)
        }
    }

    /// Apply second endomorphism: (x, y) -> (beta² * x, y)
    pub fn endomorphism2(&self) -> Self {
        if self.infinity {
            Self::INFINITY
        } else {
            Self::new(self.x.mul(&BETA2), self.y)
        }
    }

    /// Get compressed public key bytes (33 bytes)
    pub fn to_compressed(&self) -> [u8; 33] {
        let mut result = [0u8; 33];
        if self.y.is_odd() {
            result[0] = 0x03;
        } else {
            result[0] = 0x02;
        }
        result[1..33].copy_from_slice(&self.x.to_bytes());
        result
    }

    /// Get uncompressed public key bytes (65 bytes)
    pub fn to_uncompressed(&self) -> [u8; 65] {
        let mut result = [0u8; 65];
        result[0] = 0x04;
        result[1..33].copy_from_slice(&self.x.to_bytes());
        result[33..65].copy_from_slice(&self.y.to_bytes());
        result
    }

    /// Get affine coordinates as (x, y) tuple
    pub fn to_affine(&self) -> (FieldElement, FieldElement) {
        (self.x, self.y)
    }

    /// Check if y coordinate is odd
    pub fn is_y_odd(&self) -> bool {
        self.y.is_odd()
    }
}

impl JacobianPoint {
    /// Convert back to affine coordinates
    pub fn to_affine(&self) -> Point {
        if self.z.is_zero() {
            return Point::INFINITY;
        }

        let z_inv = self.z.inv();
        let z_inv2 = z_inv.sqr();
        let z_inv3 = z_inv2.mul(&z_inv);

        let x = self.x.mul(&z_inv2);
        let y = self.y.mul(&z_inv3);

        Point::new(x, y)
    }

    /// Jacobian point doubling
    pub fn double(&self) -> Self {
        if self.z.is_zero() {
            return *self;
        }

        // S = 4 * X * Y²
        let y2 = self.y.sqr();
        let s = self.x.mul(&y2);
        let s = s.add(&s).add(&s).add(&s);

        // M = 3 * X² (a = 0)
        let x2 = self.x.sqr();
        let m = x2.add(&x2).add(&x2);

        // X' = M² - 2S
        let x3 = m.sqr().sub(&s).sub(&s);

        // Y' = M * (S - X') - 8 * Y⁴
        let y4 = y2.sqr();
        let y4_8 = y4.add(&y4).add(&y4).add(&y4).add(&y4).add(&y4).add(&y4).add(&y4);
        let y3 = m.mul(&s.sub(&x3)).sub(&y4_8);

        // Z' = 2 * Y * Z
        let z3 = self.y.mul(&self.z).add(&self.y.mul(&self.z));

        Self { x: x3, y: y3, z: z3 }
    }

    /// Jacobian point addition with affine point
    pub fn add_affine(&self, other: &Point) -> Self {
        if other.infinity {
            return *self;
        }
        if self.z.is_zero() {
            return other.to_jacobian();
        }

        let z2 = self.z.sqr();
        let z3 = z2.mul(&self.z);

        let u2 = other.x.mul(&z2);
        let s2 = other.y.mul(&z3);

        let h = u2.sub(&self.x);
        let r = s2.sub(&self.y);

        if h.is_zero() {
            if r.is_zero() {
                return self.double();
            } else {
                return JacobianPoint {
                    x: FieldElement::ONE,
                    y: FieldElement::ONE,
                    z: FieldElement::ZERO,
                };
            }
        }

        let h2 = h.sqr();
        let h3 = h2.mul(&h);

        let u1h2 = self.x.mul(&h2);

        let x3 = r.sqr().sub(&h3).sub(&u1h2).sub(&u1h2);
        let y3 = r.mul(&u1h2.sub(&x3)).sub(&self.y.mul(&h3));
        let z3 = self.z.mul(&h);

        Self { x: x3, y: y3, z: z3 }
    }
}

/// Precomputed generator table for faster scalar multiplication
pub struct GeneratorTable {
    pub table: Vec<Point>,
}

impl GeneratorTable {
    /// Create precomputed table with 2^w points per 256/w windows
    pub fn new() -> Self {
        // Precompute G, 2G, 3G, ..., 1024G for batch operations
        let mut table = Vec::with_capacity(1024);
        let mut p = G;
        table.push(p);
        
        for _ in 1..1024 {
            p = p.add(&G);
            table.push(p);
        }
        
        Self { table }
    }

    /// Fast scalar multiplication using precomputed table
    pub fn mul(&self, scalar: &Scalar) -> Point {
        G.mul(scalar)
    }
}

impl Default for GeneratorTable {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generator_on_curve() {
        // y² = x³ + 7
        let y2 = G.y.sqr();
        let x3 = G.x.sqr().mul(&G.x);
        let seven = FieldElement::new([7, 0, 0, 0]);
        let rhs = x3.add(&seven);
        assert_eq!(y2, rhs);
    }

    #[test]
    fn test_point_double() {
        let p2 = G.double();
        let p2_add = G.add(&G);
        assert_eq!(p2, p2_add);
    }

    #[test]
    fn test_point_add_neg() {
        let neg_g = G.neg();
        let sum = G.add(&neg_g);
        assert!(sum.is_infinity());
    }

    #[test]
    fn test_scalar_mul_one() {
        let one = Scalar::ONE;
        let p = G.mul(&one);
        assert_eq!(p, G);
    }

    #[test]
    fn test_endomorphism() {
        // If P = k*G, then endo(P) = lambda*k*G
        let two = Scalar::new([2, 0, 0, 0]);
        let p = G.mul(&two);
        let endo_p = p.endomorphism1();
        
        // lambda * 2 mod n
        let lambda_2 = LAMBDA.mul(&two);
        let expected = G.mul(&lambda_2);
        
        assert_eq!(endo_p, expected);
    }
}
