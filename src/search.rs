use crate::config::Config;
use crate::crypto::{self, Crypto, bech32_utils};
use crate::gpu::{detect_gpu, GpuBackend};
use crate::result::{SearchResult, GpuFoundItem};
use anyhow::Result;
use std::time::Instant;

pub struct VanitySearch {
    config: Config,
    backend: Box<dyn GpuBackend>,
    crypto: Crypto,
    base_keys: Vec<[u8; 32]>,
    total_checked: u64,
    start_time: Instant,
}

impl VanitySearch {
    pub fn new(config: Config) -> Result<Self> {
        let backend = detect_gpu(config.gpu, config.backend.as_deref())?;
        let crypto = Crypto::new();
        
        // Generate starting keys
        let num_threads = backend.compute_units() as usize * 8;
        let base_keys = Self::generate_keys(&config, num_threads);
        
        Ok(Self {
            config,
            backend,
            crypto,
            base_keys,
            total_checked: 0,
            start_time: Instant::now(),
        })
    }
    
    fn generate_keys(config: &Config, count: usize) -> Vec<[u8; 32]> {
        use rand::RngCore;
        use sha2::{Sha256, Digest};
        
        let mut keys = Vec::with_capacity(count);
        
        if let Some(seed) = &config.seed {
            for i in 0..count {
                let mut key = [0u8; 32];
                let indexed_seed = format!("{}:{}", seed, i);
                key.copy_from_slice(&Sha256::digest(indexed_seed.as_bytes()));
                keys.push(key);
            }
        } else {
            let mut rng = rand::thread_rng();
            for _ in 0..count {
                let mut key = [0u8; 32];
                rng.fill_bytes(&mut key);
                keys.push(key);
            }
        }
        
        keys
    }
    
    pub fn run(&mut self) -> Result<Vec<SearchResult>> {
        println!("Device: {} ({} CUs)", self.backend.name(), self.backend.compute_units());
        println!("Searching for: {:?}", self.config.prefix);
        println!("Difficulty: {:.0}", self.config.prefix.iter()
            .map(|p| bech32_utils::difficulty(p))
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(1.0));
        println!();
        
        self.backend.set_prefixes(&self.config.prefix)?;
        self.backend.set_keys(&self.base_keys)?;
        
        let mut found = Vec::new();
        let mut last_report = Instant::now();
        
        while found.len() < self.config.max_found as usize {
            let gpu_results = self.backend.launch()?;
            
            for item in gpu_results {
                if let Some(result) = self.verify_and_create_result(&item) {
                    println!("{}", result);
                    found.push(result);
                    
                    if found.len() >= self.config.max_found as usize {
                        break;
                    }
                }
            }
            
            self.total_checked += self.backend.keys_per_launch();
            self.backend.advance_keys();
            
            // Progress report every 2 seconds
            if last_report.elapsed().as_secs() >= 2 {
                self.print_progress();
                last_report = Instant::now();
            }
        }
        
        Ok(found)
    }
    
    fn verify_and_create_result(&self, item: &GpuFoundItem) -> Option<SearchResult> {
        let base_key = &self.base_keys[item.thread_id as usize];
        let private_key = self.crypto.recover_private_key(base_key, item.increment, item.endomorphism);
        
        // Compute public key and hash160
        let pubkey = self.crypto.public_key(&private_key).ok()?;
        let hash160 = crypto::hash160(&pubkey);
        
        // Generate addresses
        let address_mainnet = bech32_utils::encode(&hash160, true);
        let address_testnet = bech32_utils::encode(&hash160, false);
        
        // Check if address matches prefix
        let matches = self.config.prefix.iter()
            .any(|p| address_mainnet.starts_with(p) || address_testnet.starts_with(p));
        
        if !matches {
            eprintln!("Warning: False positive detected");
            return None;
        }
        
        Some(SearchResult {
            address_mainnet,
            address_testnet,
            private_key_hex: hex::encode(private_key),
            wif_mainnet: self.crypto.to_wif(&private_key, true),
            wif_testnet: self.crypto.to_wif(&private_key, false),
            public_key_hex: hex::encode(pubkey),
        })
    }
    
    fn print_progress(&self) {
        let elapsed = self.start_time.elapsed().as_secs_f64();
        let rate = self.total_checked as f64 / elapsed;
        let log2 = (self.total_checked as f64).log2();
        
        print!("\r[{:.2} MKey/s] [Total: 2^{:.1}]   ", rate / 1_000_000.0, log2);
        use std::io::Write;
        std::io::stdout().flush().ok();
    }
}
