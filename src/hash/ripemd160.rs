//! RIPEMD-160 implementation

const K1: [u32; 5] = [0x00000000, 0x5a827999, 0x6ed9eba1, 0x8f1bbcdc, 0xa953fd4e];
const K2: [u32; 5] = [0x50a28be6, 0x5c4dd124, 0x6d703ef3, 0x7a6d76e9, 0x00000000];

const R1: [usize; 80] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    7, 4, 13, 1, 10, 6, 15, 3, 12, 0, 9, 5, 2, 14, 11, 8,
    3, 10, 14, 4, 9, 15, 8, 1, 2, 7, 0, 6, 13, 11, 5, 12,
    1, 9, 11, 10, 0, 8, 12, 4, 13, 3, 7, 15, 14, 5, 6, 2,
    4, 0, 5, 9, 7, 12, 2, 10, 14, 1, 3, 8, 11, 6, 15, 13,
];

const R2: [usize; 80] = [
    5, 14, 7, 0, 9, 2, 11, 4, 13, 6, 15, 8, 1, 10, 3, 12,
    6, 11, 3, 7, 0, 13, 5, 10, 14, 15, 8, 12, 4, 9, 1, 2,
    15, 5, 1, 3, 7, 14, 6, 9, 11, 8, 12, 2, 10, 0, 4, 13,
    8, 6, 4, 1, 3, 11, 15, 0, 5, 12, 2, 13, 9, 7, 10, 14,
    12, 15, 10, 4, 1, 5, 8, 7, 6, 2, 13, 14, 0, 3, 9, 11,
];

const S1: [u32; 80] = [
    11, 14, 15, 12, 5, 8, 7, 9, 11, 13, 14, 15, 6, 7, 9, 8,
    7, 6, 8, 13, 11, 9, 7, 15, 7, 12, 15, 9, 11, 7, 13, 12,
    11, 13, 6, 7, 14, 9, 13, 15, 14, 8, 13, 6, 5, 12, 7, 5,
    11, 12, 14, 15, 14, 15, 9, 8, 9, 14, 5, 6, 8, 6, 5, 12,
    9, 15, 5, 11, 6, 8, 13, 12, 5, 12, 13, 14, 11, 8, 5, 6,
];

const S2: [u32; 80] = [
    8, 9, 9, 11, 13, 15, 15, 5, 7, 7, 8, 11, 14, 14, 12, 6,
    9, 13, 15, 7, 12, 8, 9, 11, 7, 7, 12, 7, 6, 15, 13, 11,
    9, 7, 15, 11, 8, 6, 6, 14, 12, 13, 5, 14, 13, 13, 7, 5,
    15, 5, 8, 11, 14, 14, 6, 14, 6, 9, 12, 9, 12, 5, 15, 8,
    8, 5, 12, 9, 12, 5, 14, 6, 8, 13, 6, 5, 15, 13, 11, 11,
];

#[inline]
fn rotl(x: u32, n: u32) -> u32 {
    x.rotate_left(n)
}

#[inline]
fn f(j: usize, x: u32, y: u32, z: u32) -> u32 {
    match j {
        0..=15 => x ^ y ^ z,
        16..=31 => (x & y) | (!x & z),
        32..=47 => (x | !y) ^ z,
        48..=63 => (x & z) | (y & !z),
        _ => x ^ (y | !z),
    }
}

/// RIPEMD-160 hash context
pub struct Ripemd160 {
    state: [u32; 5],
    buffer: [u8; 64],
    buffer_len: usize,
    total_len: u64,
}

impl Ripemd160 {
    pub fn new() -> Self {
        Self {
            state: [0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476, 0xc3d2e1f0],
            buffer: [0u8; 64],
            buffer_len: 0,
            total_len: 0,
        }
    }

    fn transform(&mut self, block: &[u8]) {
        let mut w = [0u32; 16];
        for i in 0..16 {
            w[i] = u32::from_le_bytes([
                block[i * 4],
                block[i * 4 + 1],
                block[i * 4 + 2],
                block[i * 4 + 3],
            ]);
        }

        let mut a1 = self.state[0];
        let mut b1 = self.state[1];
        let mut c1 = self.state[2];
        let mut d1 = self.state[3];
        let mut e1 = self.state[4];

        let mut a2 = self.state[0];
        let mut b2 = self.state[1];
        let mut c2 = self.state[2];
        let mut d2 = self.state[3];
        let mut e2 = self.state[4];

        for j in 0..80 {
            let round = j / 16;
            
            // Left path
            let t = rotl(
                a1.wrapping_add(f(j, b1, c1, d1))
                    .wrapping_add(w[R1[j]])
                    .wrapping_add(K1[round]),
                S1[j],
            )
            .wrapping_add(e1);
            a1 = e1;
            e1 = d1;
            d1 = rotl(c1, 10);
            c1 = b1;
            b1 = t;

            // Right path
            let t = rotl(
                a2.wrapping_add(f(79 - j, b2, c2, d2))
                    .wrapping_add(w[R2[j]])
                    .wrapping_add(K2[round]),
                S2[j],
            )
            .wrapping_add(e2);
            a2 = e2;
            e2 = d2;
            d2 = rotl(c2, 10);
            c2 = b2;
            b2 = t;
        }

        let t = self.state[1].wrapping_add(c1).wrapping_add(d2);
        self.state[1] = self.state[2].wrapping_add(d1).wrapping_add(e2);
        self.state[2] = self.state[3].wrapping_add(e1).wrapping_add(a2);
        self.state[3] = self.state[4].wrapping_add(a1).wrapping_add(b2);
        self.state[4] = self.state[0].wrapping_add(b1).wrapping_add(c2);
        self.state[0] = t;
    }

    pub fn update(&mut self, data: &[u8]) {
        self.total_len += data.len() as u64;
        let mut offset = 0;

        if self.buffer_len > 0 {
            let to_copy = std::cmp::min(64 - self.buffer_len, data.len());
            self.buffer[self.buffer_len..self.buffer_len + to_copy]
                .copy_from_slice(&data[..to_copy]);
            self.buffer_len += to_copy;
            offset = to_copy;

            if self.buffer_len == 64 {
                let block = self.buffer;
                self.transform(&block);
                self.buffer_len = 0;
            }
        }

        while offset + 64 <= data.len() {
            self.transform(&data[offset..offset + 64]);
            offset += 64;
        }

        if offset < data.len() {
            let remaining = data.len() - offset;
            self.buffer[..remaining].copy_from_slice(&data[offset..]);
            self.buffer_len = remaining;
        }
    }

    pub fn finalize(mut self) -> [u8; 20] {
        let bit_len = self.total_len * 8;

        self.buffer[self.buffer_len] = 0x80;
        self.buffer_len += 1;

        if self.buffer_len > 56 {
            while self.buffer_len < 64 {
                self.buffer[self.buffer_len] = 0;
                self.buffer_len += 1;
            }
            let block = self.buffer;
            self.transform(&block);
            self.buffer_len = 0;
        }

        while self.buffer_len < 56 {
            self.buffer[self.buffer_len] = 0;
            self.buffer_len += 1;
        }

        self.buffer[56..64].copy_from_slice(&bit_len.to_le_bytes());
        let block = self.buffer;
        self.transform(&block);

        let mut result = [0u8; 20];
        for (i, &word) in self.state.iter().enumerate() {
            result[i * 4..i * 4 + 4].copy_from_slice(&word.to_le_bytes());
        }
        result
    }
}

impl Default for Ripemd160 {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute RIPEMD-160 hash of data
pub fn ripemd160(data: &[u8]) -> [u8; 20] {
    let mut hasher = Ripemd160::new();
    hasher.update(data);
    hasher.finalize()
}

/// Optimized RIPEMD-160 for exactly 32-byte SHA256 hash output
/// This is ~2x faster than generic ripemd160() for this specific size
#[inline]
pub fn ripemd160_32(data: &[u8; 32]) -> [u8; 20] {
    // Pre-build the padded 64-byte block:
    // bytes 0-31: data (32 bytes)
    // byte 32: 0x80 (padding start)
    // bytes 33-55: zeros
    // bytes 56-63: bit length in LE (32*8 = 256 = 0x100)
    
    let mut block = [0u8; 64];
    block[0..32].copy_from_slice(data);
    block[32] = 0x80;
    // bytes 33-55 are already zeros
    // Length in bits: 32 * 8 = 256 = 0x0000000000000100 (little-endian)
    block[56] = 0x00;
    block[57] = 0x01;
    // bytes 58-63 are already zeros
    
    // Prepare message words
    let mut w = [0u32; 16];
    for i in 0..16 {
        w[i] = u32::from_le_bytes([
            block[i * 4],
            block[i * 4 + 1],
            block[i * 4 + 2],
            block[i * 4 + 3],
        ]);
    }
    
    // Initial state
    let mut state = [0x67452301u32, 0xefcdab89, 0x98badcfe, 0x10325476, 0xc3d2e1f0];
    
    let mut a1 = state[0];
    let mut b1 = state[1];
    let mut c1 = state[2];
    let mut d1 = state[3];
    let mut e1 = state[4];

    let mut a2 = state[0];
    let mut b2 = state[1];
    let mut c2 = state[2];
    let mut d2 = state[3];
    let mut e2 = state[4];

    for j in 0..80 {
        let round = j / 16;
        
        // Left path
        let t = rotl(
            a1.wrapping_add(f(j, b1, c1, d1))
                .wrapping_add(w[R1[j]])
                .wrapping_add(K1[round]),
            S1[j],
        )
        .wrapping_add(e1);
        a1 = e1;
        e1 = d1;
        d1 = rotl(c1, 10);
        c1 = b1;
        b1 = t;

        // Right path
        let t = rotl(
            a2.wrapping_add(f(79 - j, b2, c2, d2))
                .wrapping_add(w[R2[j]])
                .wrapping_add(K2[round]),
            S2[j],
        )
        .wrapping_add(e2);
        a2 = e2;
        e2 = d2;
        d2 = rotl(c2, 10);
        c2 = b2;
        b2 = t;
    }

    let t = state[1].wrapping_add(c1).wrapping_add(d2);
    state[1] = state[2].wrapping_add(d1).wrapping_add(e2);
    state[2] = state[3].wrapping_add(e1).wrapping_add(a2);
    state[3] = state[4].wrapping_add(a1).wrapping_add(b2);
    state[4] = state[0].wrapping_add(b1).wrapping_add(c2);
    state[0] = t;
    
    // Output
    let mut result = [0u8; 20];
    for (i, &word) in state.iter().enumerate() {
        result[i * 4..i * 4 + 4].copy_from_slice(&word.to_le_bytes());
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ripemd160_empty() {
        let hash = ripemd160(b"");
        let expected = hex::decode("9c1185a5c5e9fc54612808977ee8f548b2258d31").unwrap();
        assert_eq!(&hash[..], &expected[..]);
    }

    #[test]
    fn test_ripemd160_abc() {
        let hash = ripemd160(b"abc");
        let expected = hex::decode("8eb208f7e05d987a9b044a8e98c6b087f15a0bfc").unwrap();
        assert_eq!(&hash[..], &expected[..]);
    }

    #[test]
    fn test_ripemd160_message() {
        let hash = ripemd160(b"message digest");
        let expected = hex::decode("5d0689ef49d2fae572b881b123a85ffa21595f36").unwrap();
        assert_eq!(&hash[..], &expected[..]);
    }
}
