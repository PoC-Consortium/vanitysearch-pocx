//! Output formatting for vanity search results

use crate::search::Match;
use crate::wif::{create_descriptor, encode_wif};

/// Network info for output formatting
#[derive(Debug, Clone)]
pub struct NetworkInfo {
    pub name: String,
    pub is_mainnet: bool,
}

impl Default for NetworkInfo {
    fn default() -> Self {
        Self {
            name: "POCX".to_string(),
            is_mainnet: true,
        }
    }
}

/// Formatted output for a match
#[derive(Debug, Clone)]
pub struct FormattedMatch {
    pub address: String,
    pub descriptors: Vec<(String, String)>, // (network_name, descriptor)
    pub private_key_hex: String,
}

impl FormattedMatch {
    pub fn from_match(m: &Match, networks: &[NetworkInfo]) -> Self {
        let mut descriptors = Vec::new();

        for net in networks {
            let wif = encode_wif(&m.private_key, m.compressed, net.is_mainnet);
            let descriptor = create_descriptor(&wif);
            descriptors.push((net.name.clone(), descriptor));
        }

        let private_key_hex = format!("0x{}", hex::encode(m.private_key.to_bytes()));

        Self {
            address: m.address.clone(),
            descriptors,
            private_key_hex,
        }
    }

    /// Format as human-readable text
    pub fn to_text(&self) -> String {
        self.to_text_colored(false)
    }

    /// Format as human-readable text with optional ANSI colors
    pub fn to_text_colored(&self, color: bool) -> String {
        // ANSI color codes
        let green = if color { "\x1b[32m" } else { "" };
        let yellow = if color { "\x1b[33m" } else { "" };
        let red = if color { "\x1b[31m" } else { "" };
        let reset = if color { "\x1b[0m" } else { "" };

        let mut lines = vec![format!("Address \t{}{}{}", green, self.address, reset)];

        for (name, desc) in &self.descriptors {
            // Color the WIF inside the descriptor yellow
            // Descriptor format: wpkh(WIF)#checksum
            let colored_desc = if color {
                if let (Some(start), Some(end)) = (desc.find('('), desc.find(')')) {
                    let prefix = &desc[..=start];
                    let wif = &desc[start + 1..end];
                    let suffix = &desc[end..];
                    format!("{}{}{}{}", prefix, yellow, wif, reset).to_string() + suffix
                } else {
                    desc.clone()
                }
            } else {
                desc.clone()
            };
            lines.push(format!("{} Descriptor\t{}", name, colored_desc));
        }

        lines.push(format!(
            "Private Key \t{}{}{}",
            red, self.private_key_hex, reset
        ));

        lines.join("\n")
    }

    /// Format as JSON
    pub fn to_json(&self) -> String {
        let descriptors_json: Vec<String> = self
            .descriptors
            .iter()
            .map(|(name, desc)| format!(r#"{{"network":"{}","descriptor":"{}"}}"#, name, desc))
            .collect();

        format!(
            r#"{{"address":"{}","descriptors":[{}],"private_key_hex":"{}"}}"#,
            self.address,
            descriptors_json.join(","),
            self.private_key_hex
        )
    }

    /// Format as CSV line
    pub fn to_csv(&self) -> String {
        let descriptors_str: Vec<String> = self
            .descriptors
            .iter()
            .map(|(name, desc)| format!("{}:{}", name, desc))
            .collect();

        format!(
            "{},{},{}",
            self.address,
            descriptors_str.join(";"),
            self.private_key_hex
        )
    }
}

/// Format statistics
pub struct Stats {
    pub keys_per_second: f64,
    pub total_keys: u64,
    pub matches_found: u64,
    pub elapsed_secs: f64,
}

impl Stats {
    pub fn format(&self) -> String {
        let keys_str = if self.keys_per_second >= 1_000_000_000.0 {
            format!("{:.2} Gkey/s", self.keys_per_second / 1_000_000_000.0)
        } else if self.keys_per_second >= 1_000_000.0 {
            format!("{:.2} Mkey/s", self.keys_per_second / 1_000_000.0)
        } else if self.keys_per_second >= 1_000.0 {
            format!("{:.2} Kkey/s", self.keys_per_second / 1_000.0)
        } else {
            format!("{:.2} key/s", self.keys_per_second)
        };

        format!(
            "[{:.1}s] {} | {} keys checked | {} matches",
            self.elapsed_secs, keys_str, self.total_keys, self.matches_found
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::secp256k1::Scalar;

    #[test]
    fn test_formatted_match() {
        let key = Scalar::new([1, 0, 0, 0]);
        let m = Match {
            address: "bc1qtest".to_string(),
            private_key: key,
            compressed: true,
        };

        let networks = vec![
            NetworkInfo {
                name: "Main".to_string(),
                is_mainnet: true,
            },
            NetworkInfo {
                name: "Test".to_string(),
                is_mainnet: false,
            },
        ];

        let formatted = FormattedMatch::from_match(&m, &networks);
        assert_eq!(formatted.descriptors.len(), 2);
        assert!(formatted.descriptors[0].1.contains('#'));
        assert!(formatted.descriptors[1].1.contains('#'));
    }
}
