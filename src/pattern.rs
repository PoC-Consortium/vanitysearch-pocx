//! Pattern matching for vanity addresses
//! Supports wildcards: ? (any single character), * (any sequence)

/// Bech32 charset reverse lookup (char -> 5-bit value)
const BECH32_REV: [i8; 128] = [
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    15, -1, 10, 17, 21, 20, 26, 30, 7, 5, -1, -1, -1, -1, -1, -1, // 0-9
    -1, 29, -1, 24, 13, 25, 9, 8, 23, -1, 18, 22, 31, 27, 19, -1, // A-O
    1, 0, 3, 16, 11, 28, 12, 14, 6, 4, 2, -1, -1, -1, -1, -1, // P-Z
    -1, 29, -1, 24, 13, 25, 9, 8, 23, -1, 18, 22, 31, 27, 19, -1, // a-o
    1, 0, 3, 16, 11, 28, 12, 14, 6, 4, 2, -1, -1, -1, -1, -1, // p-z
];

/// Fast prefix matcher for hash160 bytes
/// Pre-computes the pattern as 5-bit values for direct matching
#[derive(Debug, Clone)]
pub struct FastPrefixMatcher {
    /// The 5-bit values to match (after witness version 'q')
    pub prefix_5bit: Vec<u8>,
    /// Number of 5-bit values to match
    pub prefix_len: usize,
    /// Whether the pattern has wildcards (requires full matching)
    pub has_wildcards: bool,
}

impl FastPrefixMatcher {
    /// Create from a data pattern (part after "hrp1")
    pub fn new(data_pattern: &str) -> Self {
        let chars: Vec<char> = data_pattern.chars().collect();

        // Check for wildcards
        let has_wildcards = chars.iter().any(|&c| c == '?' || c == '*');

        if has_wildcards {
            // Find the longest contiguous concrete segment (no wildcards)
            // We need to find a good prefix to filter on, even if pattern starts with *
            // E.g., "q*ev?seed*" should use "ev" or "seed" as filter prefix

            // First, try prefix before first wildcard (skip witness version 'q')
            let prefix_end = chars
                .iter()
                .skip(1) // Skip 'q'
                .position(|&c| c == '?' || c == '*')
                .map(|p| p + 1) // Adjust for skip
                .unwrap_or(chars.len());

            let prefix_chars: Vec<char> = if prefix_end > 1 {
                chars[1..prefix_end].to_vec()
            } else {
                vec![]
            };

            let prefix_5bit: Vec<u8> = prefix_chars
                .iter()
                .filter_map(|&c| {
                    let idx = c as usize;
                    if idx < 128 && BECH32_REV[idx] >= 0 {
                        Some(BECH32_REV[idx] as u8)
                    } else {
                        None
                    }
                })
                .collect();

            // Store info about where the prefix starts in the pattern
            // (for patterns like "q*evlseed", we can't use fast hash160 matching
            // because the segment isn't at a fixed position)
            let prefix_at_start = prefix_end > 1;

            Self {
                prefix_len: if prefix_at_start {
                    prefix_5bit.len()
                } else {
                    0
                },
                prefix_5bit: if prefix_at_start { prefix_5bit } else { vec![] },
                has_wildcards,
            }
        } else {
            // No wildcards - convert entire prefix (skip witness version 'q')
            let prefix_5bit: Vec<u8> = if chars.len() > 1 {
                chars[1..]
                    .iter()
                    .filter_map(|&c| {
                        let idx = c as usize;
                        if idx < 128 && BECH32_REV[idx] >= 0 {
                            Some(BECH32_REV[idx] as u8)
                        } else {
                            None
                        }
                    })
                    .collect()
            } else {
                vec![]
            };

            Self {
                prefix_len: prefix_5bit.len(),
                prefix_5bit,
                has_wildcards,
            }
        }
    }

    /// Check if hash160 matches the prefix pattern
    /// Returns true if the bech32-encoded hash160 would start with the pattern
    #[inline]
    pub fn matches_hash160(&self, hash160: &[u8; 20]) -> bool {
        if self.prefix_len == 0 {
            return true;
        }

        // Convert hash160 to 5-bit values on the fly
        // We only need to convert enough bits to check the prefix
        let bits_needed = self.prefix_len * 5;
        let bytes_needed = bits_needed.div_ceil(8);

        // Extract 5-bit values from hash160
        let mut acc: u32 = 0;
        let mut bits: u32 = 0;
        let mut idx = 0;

        for &byte in hash160.iter().take(bytes_needed.min(20)) {
            acc = (acc << 8) | (byte as u32);
            bits += 8;

            while bits >= 5 && idx < self.prefix_len {
                bits -= 5;
                let val = ((acc >> bits) & 0x1f) as u8;
                if val != self.prefix_5bit[idx] {
                    return false;
                }
                idx += 1;
            }
        }

        true
    }
}

/// Pattern matcher for bech32 addresses
#[derive(Debug, Clone)]
pub struct Pattern {
    /// The pattern string (lowercase)
    pub pattern: String,
    /// HRP extracted from pattern (e.g., "bc", "tb")
    pub hrp: String,
    /// Data part of pattern (after hrp1)
    pub data_pattern: String,
    /// Whether this is a full address pattern
    pub is_full: bool,
    /// Estimated difficulty (2^bits)
    pub difficulty: f64,
    /// Fast prefix matcher for hash160 matching
    pub fast_matcher: FastPrefixMatcher,
}

impl Pattern {
    /// Create a new pattern from string
    pub fn new(pattern: &str) -> Result<Self, PatternError> {
        let pattern = pattern.to_lowercase();

        // Validate pattern format
        if pattern.len() < 4 {
            return Err(PatternError::TooShort);
        }

        // Find HRP separator
        let sep_pos = pattern.find('1').ok_or(PatternError::NoSeparator)?;
        if sep_pos < 1 {
            return Err(PatternError::InvalidHrp);
        }

        let hrp = pattern[..sep_pos].to_string();
        let data_pattern = pattern[sep_pos + 1..].to_string();

        // Validate HRP
        if !hrp.chars().all(|c| c.is_ascii_lowercase()) {
            return Err(PatternError::InvalidHrp);
        }

        // Check witness version (should be 'q' for version 0)
        if !data_pattern.is_empty() {
            let first_char = data_pattern.chars().next().unwrap();
            if first_char != 'q' && first_char != '?' && first_char != '*' {
                return Err(PatternError::InvalidWitnessVersion);
            }
        }

        // Validate data pattern characters
        for c in data_pattern.chars() {
            if !is_valid_pattern_char(c) {
                return Err(PatternError::InvalidCharacter(c));
            }
        }

        // Calculate difficulty
        // Count concrete characters EXCLUDING the witness version 'q' at position 0
        // since 'q' is always present for bech32 witness v0 addresses
        let concrete_chars = data_pattern
            .chars()
            .skip(1) // Skip witness version 'q'
            .filter(|&c| c != '?' && c != '*')
            .count();

        // Each bech32 character is 5 bits (32 possible values)
        let difficulty = (32.0f64).powi(concrete_chars as i32);

        // Check if it's a full 42-character address (hrp + 1 + 39 data + 6 checksum)
        // For bc: 2 + 1 + 39 = 42
        let is_full =
            data_pattern.len() >= 39 && !data_pattern.contains('?') && !data_pattern.contains('*');

        // Create fast prefix matcher
        let fast_matcher = FastPrefixMatcher::new(&data_pattern);

        Ok(Self {
            pattern,
            hrp,
            data_pattern: data_pattern.clone(),
            is_full,
            difficulty,
            fast_matcher,
        })
    }

    /// Fast check if a hash160 matches the prefix pattern
    /// This avoids the full bech32 encoding when possible
    #[inline]
    pub fn matches_hash160(&self, hash160: &[u8; 20]) -> bool {
        self.fast_matcher.matches_hash160(hash160)
    }

    /// Check if an address matches this pattern
    pub fn matches(&self, address: &str) -> bool {
        let address = address.to_lowercase();

        // Check HRP
        if !address.starts_with(&self.hrp) {
            return false;
        }

        // Check separator
        let expected_sep = self.hrp.len();
        if address.len() <= expected_sep || address.chars().nth(expected_sep) != Some('1') {
            return false;
        }

        // Get data part of address
        let addr_data = &address[expected_sep + 1..];

        // Match pattern
        match_wildcard(&self.data_pattern, addr_data)
    }

    /// Get pattern prefix (characters before first wildcard)
    pub fn prefix(&self) -> &str {
        let end = self
            .data_pattern
            .find(['?', '*'])
            .unwrap_or(self.data_pattern.len());
        &self.data_pattern[..end]
    }

    /// Get minimum prefix length for matching
    pub fn min_prefix_len(&self) -> usize {
        self.prefix().len()
    }
}

/// Check if character is valid in a pattern
#[inline]
fn is_valid_pattern_char(c: char) -> bool {
    // Bech32 charset + wildcards
    matches!(
        c,
        'q' | 'p'
            | 'z'
            | 'r'
            | 'y'
            | '9'
            | 'x'
            | '8'
            | 'g'
            | 'f'
            | '2'
            | 't'
            | 'v'
            | 'd'
            | 'w'
            | '0'
            | 's'
            | '3'
            | 'j'
            | 'n'
            | '5'
            | '4'
            | 'k'
            | 'h'
            | 'c'
            | 'e'
            | '6'
            | 'm'
            | 'u'
            | 'a'
            | '7'
            | 'l'
            | '?'
            | '*'
    )
}

/// Match string against pattern with wildcards
/// If pattern doesn't end with *, it's treated as a prefix match
fn match_wildcard(pattern: &str, string: &str) -> bool {
    let pattern: Vec<char> = pattern.chars().collect();
    let string: Vec<char> = string.chars().collect();

    // If pattern doesn't contain any wildcards, do simple prefix match
    let has_wildcards = pattern.iter().any(|&c| c == '?' || c == '*');

    if !has_wildcards {
        // Pure prefix matching
        if string.len() < pattern.len() {
            return false;
        }
        return pattern.iter().zip(string.iter()).all(|(p, s)| p == s);
    }

    // If pattern ends with *, use wildcard matching as-is
    // Otherwise, append implicit * for prefix behavior
    if pattern.last() == Some(&'*') {
        match_wildcard_recursive(&pattern, &string, 0, 0)
    } else {
        let mut pattern_with_star = pattern;
        pattern_with_star.push('*');
        match_wildcard_recursive(&pattern_with_star, &string, 0, 0)
    }
}

fn match_wildcard_recursive(pattern: &[char], string: &[char], pi: usize, si: usize) -> bool {
    // Base cases
    if pi == pattern.len() && si == string.len() {
        return true;
    }

    if pi == pattern.len() {
        return false;
    }

    let pc = pattern[pi];

    match pc {
        '*' => {
            // Try matching * with 0 or more characters
            // First, try matching rest of pattern with current position
            if match_wildcard_recursive(pattern, string, pi + 1, si) {
                return true;
            }
            // Then, try consuming one character and continue with *
            if si < string.len() && match_wildcard_recursive(pattern, string, pi, si + 1) {
                return true;
            }
            false
        }
        '?' => {
            // Match any single character
            if si < string.len() {
                match_wildcard_recursive(pattern, string, pi + 1, si + 1)
            } else {
                false
            }
        }
        _ => {
            // Match exact character
            if si < string.len() && string[si] == pc {
                match_wildcard_recursive(pattern, string, pi + 1, si + 1)
            } else {
                false
            }
        }
    }
}

/// Fast prefix matching (for GPU filter)
#[inline]
pub fn matches_prefix(addr_data: &str, pattern_prefix: &str) -> bool {
    if addr_data.len() < pattern_prefix.len() {
        return false;
    }
    addr_data[..pattern_prefix.len()] == *pattern_prefix
}

/// Convert hash160 to bech32 address data (5-bit values)
pub fn hash160_to_bech32_data(hash160: &[u8; 20]) -> Vec<u8> {
    let mut data = Vec::with_capacity(33);
    data.push(0); // witness version 0

    // Convert 8-bit to 5-bit
    let mut acc = 0u32;
    let mut bits = 0;

    for &byte in hash160 {
        acc = (acc << 8) | (byte as u32);
        bits += 8;
        while bits >= 5 {
            bits -= 5;
            data.push(((acc >> bits) & 0x1f) as u8);
        }
    }

    if bits > 0 {
        data.push(((acc << (5 - bits)) & 0x1f) as u8);
    }

    data
}

#[derive(Debug, Clone)]
pub enum PatternError {
    TooShort,
    NoSeparator,
    InvalidHrp,
    InvalidWitnessVersion,
    InvalidCharacter(char),
}

impl std::fmt::Display for PatternError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PatternError::TooShort => write!(f, "Pattern too short (min 4 chars)"),
            PatternError::NoSeparator => write!(f, "Missing separator '1'"),
            PatternError::InvalidHrp => write!(f, "Invalid HRP"),
            PatternError::InvalidWitnessVersion => {
                write!(f, "Invalid witness version (expected 'q' for v0)")
            }
            PatternError::InvalidCharacter(c) => write!(f, "Invalid character in pattern: '{}'", c),
        }
    }
}

impl std::error::Error for PatternError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pattern_parse() {
        let pattern = Pattern::new("bc1qmadf0*").unwrap();
        assert_eq!(pattern.hrp, "bc");
        assert_eq!(pattern.data_pattern, "qmadf0*");
        assert!(!pattern.is_full);
    }

    #[test]
    fn test_pattern_match_exact() {
        let pattern = Pattern::new("bc1qtest").unwrap();
        assert!(pattern.matches("bc1qtestzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz"));
        assert!(!pattern.matches("bc1qwrongzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz"));
    }

    #[test]
    fn test_pattern_match_wildcard_question() {
        let pattern = Pattern::new("bc1q?est").unwrap();
        assert!(pattern.matches("bc1qtestzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz"));
        assert!(pattern.matches("bc1qaestzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz"));
        assert!(!pattern.matches("bc1qteszzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz"));
    }

    #[test]
    fn test_pattern_match_wildcard_star() {
        let pattern = Pattern::new("bc1qmadf0*").unwrap();
        assert!(pattern.matches("bc1qmadf0zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz"));
        assert!(pattern.matches("bc1qmadf0abcdefzzzzzzzzzzzzzzzzzzzzzzzz"));
        assert!(!pattern.matches("bc1qmadf1zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz"));
    }

    #[test]
    fn test_pattern_prefix() {
        let pattern = Pattern::new("bc1qmadf0*").unwrap();
        assert_eq!(pattern.prefix(), "qmadf0");
    }

    #[test]
    fn test_invalid_pattern() {
        assert!(Pattern::new("bc").is_err());
        assert!(Pattern::new("bctest").is_err()); // no separator
        assert!(Pattern::new("BC1qtest").is_ok()); // should lowercase
    }

    #[test]
    fn test_difficulty() {
        let pattern = Pattern::new("bc1qmadf0*").unwrap();
        // 6 concrete chars (qmadf0), each 5 bits = 32 possibilities
        assert!(pattern.difficulty > 1.0);
    }
}
