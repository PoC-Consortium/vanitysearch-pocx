//! Output formatting for vanity search results

use crate::search::Match;
use crate::wif::{encode_wif, create_descriptor};

/// Formatted output for a match
#[derive(Debug, Clone)]
pub struct FormattedMatch {
    pub address: String,
    pub mainnet_wif: String,
    pub testnet_wif: String,
    pub mainnet_descriptor: String,
    pub testnet_descriptor: String,
    pub private_key_hex: String,
}

impl FormattedMatch {
    pub fn from_match(m: &Match) -> Self {
        let mainnet_wif = encode_wif(&m.private_key, m.compressed, true);
        let testnet_wif = encode_wif(&m.private_key, m.compressed, false);
        
        let mainnet_descriptor = create_descriptor(&mainnet_wif);
        let testnet_descriptor = create_descriptor(&testnet_wif);
        
        let private_key_hex = format!("0x{}", hex::encode(m.private_key.to_bytes()));
        
        Self {
            address: m.address.clone(),
            mainnet_wif,
            testnet_wif,
            mainnet_descriptor,
            testnet_descriptor,
            private_key_hex,
        }
    }

    /// Format as human-readable text
    pub fn to_text(&self) -> String {
        format!(
            "Address: {}\n\
             Mainnet WIF: {}\n\
             Testnet WIF: {}\n\
             Mainnet Descriptor: {}\n\
             Testnet Descriptor: {}\n\
             Private Key (hex): {}",
            self.address,
            self.mainnet_wif,
            self.testnet_wif,
            self.mainnet_descriptor,
            self.testnet_descriptor,
            self.private_key_hex
        )
    }

    /// Format as JSON
    pub fn to_json(&self) -> String {
        format!(
            r#"{{"address":"{}","mainnet_wif":"{}","testnet_wif":"{}","mainnet_descriptor":"{}","testnet_descriptor":"{}","private_key_hex":"{}"}}"#,
            self.address,
            self.mainnet_wif,
            self.testnet_wif,
            self.mainnet_descriptor,
            self.testnet_descriptor,
            self.private_key_hex
        )
    }

    /// Format as CSV line
    pub fn to_csv(&self) -> String {
        format!(
            "{},{},{},{},{},{}",
            self.address,
            self.mainnet_wif,
            self.testnet_wif,
            self.mainnet_descriptor,
            self.testnet_descriptor,
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
            self.elapsed_secs,
            keys_str,
            self.total_keys,
            self.matches_found
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
        
        let formatted = FormattedMatch::from_match(&m);
        assert!(!formatted.mainnet_wif.is_empty());
        assert!(!formatted.testnet_wif.is_empty());
        assert!(formatted.mainnet_descriptor.contains('#'));
        assert!(formatted.testnet_descriptor.contains('#'));
    }
}
