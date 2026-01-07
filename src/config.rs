//! Network configuration

use serde::{Deserialize, Serialize};
use std::path::Path;

/// Network definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Network {
    pub name: String,
    pub hrp: String,
    pub is_mainnet: bool,
}

/// Networks configuration file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworksConfig {
    pub networks: Vec<Network>,
}

impl Default for NetworksConfig {
    fn default() -> Self {
        Self {
            networks: vec![
                Network {
                    name: "Bitcoin Mainnet".to_string(),
                    hrp: "bc".to_string(),
                    is_mainnet: true,
                },
                Network {
                    name: "Bitcoin Testnet".to_string(),
                    hrp: "tb".to_string(),
                    is_mainnet: false,
                },
                Network {
                    name: "Bitcoin Signet".to_string(),
                    hrp: "tb".to_string(),
                    is_mainnet: false,
                },
                Network {
                    name: "Bitcoin Regtest".to_string(),
                    hrp: "bcrt".to_string(),
                    is_mainnet: false,
                },
            ],
        }
    }
}

impl NetworksConfig {
    /// Load configuration from file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, ConfigError> {
        let content = std::fs::read_to_string(path)?;
        let config: NetworksConfig = toml::from_str(&content)?;
        Ok(config)
    }

    /// Save configuration to file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), ConfigError> {
        let content = toml::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Find network by HRP
    pub fn find_by_hrp(&self, hrp: &str) -> Option<&Network> {
        self.networks.iter().find(|n| n.hrp == hrp)
    }

    /// Find network by name
    pub fn find_by_name(&self, name: &str) -> Option<&Network> {
        self.networks.iter().find(|n| n.name.to_lowercase() == name.to_lowercase())
    }

    /// Get mainnet networks
    pub fn mainnets(&self) -> impl Iterator<Item = &Network> {
        self.networks.iter().filter(|n| n.is_mainnet)
    }

    /// Get testnet networks
    pub fn testnets(&self) -> impl Iterator<Item = &Network> {
        self.networks.iter().filter(|n| !n.is_mainnet)
    }

    /// Create default config file if it doesn't exist
    pub fn create_default_if_missing<P: AsRef<Path>>(path: P) -> Result<Self, ConfigError> {
        if path.as_ref().exists() {
            Self::load(path)
        } else {
            let config = Self::default();
            config.save(&path)?;
            Ok(config)
        }
    }
}

#[derive(Debug)]
pub enum ConfigError {
    Io(std::io::Error),
    Parse(toml::de::Error),
    Serialize(toml::ser::Error),
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigError::Io(e) => write!(f, "IO error: {}", e),
            ConfigError::Parse(e) => write!(f, "Parse error: {}", e),
            ConfigError::Serialize(e) => write!(f, "Serialize error: {}", e),
        }
    }
}

impl std::error::Error for ConfigError {}

impl From<std::io::Error> for ConfigError {
    fn from(e: std::io::Error) -> Self {
        ConfigError::Io(e)
    }
}

impl From<toml::de::Error> for ConfigError {
    fn from(e: toml::de::Error) -> Self {
        ConfigError::Parse(e)
    }
}

impl From<toml::ser::Error> for ConfigError {
    fn from(e: toml::ser::Error) -> Self {
        ConfigError::Serialize(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = NetworksConfig::default();
        assert!(!config.networks.is_empty());
        
        let mainnet = config.find_by_hrp("bc").unwrap();
        assert!(mainnet.is_mainnet);
        assert_eq!(mainnet.hrp, "bc");
    }

    #[test]
    fn test_serialize_deserialize() {
        let config = NetworksConfig::default();
        let serialized = toml::to_string(&config).unwrap();
        let deserialized: NetworksConfig = toml::from_str(&serialized).unwrap();
        
        assert_eq!(config.networks.len(), deserialized.networks.len());
    }
}
