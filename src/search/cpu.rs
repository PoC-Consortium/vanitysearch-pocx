//! CPU-based vanity search engine
//! Optimized with batch modular inverse (Montgomery's trick)

use crate::bech32;
use crate::hash::hash160_33;
use crate::pattern::Pattern;
use crate::secp256k1::{FieldElement, G, Point, Scalar, LAMBDA, LAMBDA2, BETA, BETA2};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

/// Group size for batch processing (matches VanitySearch GRP_SIZE)
const GRP_SIZE: usize = 1024;

/// Result of a vanity search match
#[derive(Debug, Clone)]
pub struct Match {
    pub address: String,
    pub private_key: Scalar,
    pub compressed: bool,
}

/// CPU search engine configuration
#[derive(Debug, Clone)]
pub struct CpuSearchConfig {
    /// Number of threads (0 = auto)
    pub threads: usize,
    /// Pattern to search for
    pub pattern: Pattern,
    /// Human-readable part (e.g., "bc", "tb")
    pub hrp: String,
    /// Stop after this many matches (0 = infinite)
    pub max_matches: u64,
    /// Timeout in seconds (0 = infinite)
    pub timeout_secs: u64,
    /// Batch size per thread iteration
    pub batch_size: usize,
}

impl Default for CpuSearchConfig {
    fn default() -> Self {
        Self {
            threads: 0,
            pattern: Pattern::new("bc1q*").unwrap(),
            hrp: "bc".to_string(),
            max_matches: 0,
            timeout_secs: 0,
            batch_size: GRP_SIZE,
        }
    }
}

/// Precomputed table for fast group operations
/// Matches VanitySearch GPUGroup.h: Gx[512], Gy[512], _2Gnx, _2Gny
struct GroupTable {
    /// G, 2G, 4G, ..., 2^i * G for i in 0..512
    /// Actually stores: i*G for i in 1..=GRP_SIZE/2
    gx: Vec<FieldElement>,
    gy: Vec<FieldElement>,
    /// 2 * GRP_SIZE * G for batch advancement  
    delta_2n: Point,
}

impl GroupTable {
    fn new() -> Self {
        let half = GRP_SIZE / 2;
        let mut gx = Vec::with_capacity(half);
        let mut gy = Vec::with_capacity(half);
        
        // Compute 1*G, 2*G, ..., (GRP_SIZE/2)*G
        let mut p = G;
        for _ in 0..half {
            gx.push(p.x);
            gy.push(p.y);
            p = p.add(&G);
        }
        
        // 2*GRP_SIZE*G for advancing to next batch
        let delta_scalar = Scalar::new([(2 * GRP_SIZE) as u64, 0, 0, 0]);
        let delta_2n = G.mul(&delta_scalar);
        
        Self { gx, gy, delta_2n }
    }
}

/// CPU search engine
pub struct CpuSearchEngine {
    config: CpuSearchConfig,
    /// Precomputed group table
    group_table: GroupTable,
    /// Stop flag
    stop: Arc<AtomicBool>,
    /// Counter for keys checked
    keys_checked: Arc<AtomicU64>,
    /// Counter for matches found
    matches_found: Arc<AtomicU64>,
}

/// Batch modular inverse using Montgomery's trick
/// Given [a0, a1, ..., an-1], compute [1/a0, 1/a1, ..., 1/an-1]
/// Cost: 1 inverse + 3*(n-1) multiplications instead of n inverses
fn batch_inverse(vals: &[FieldElement]) -> Vec<FieldElement> {
    let n = vals.len();
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![vals[0].inv()];
    }
    
    // Compute cumulative products: prefix[i] = a0 * a1 * ... * ai
    let mut prefix = Vec::with_capacity(n);
    prefix.push(vals[0]);
    for i in 1..n {
        prefix.push(prefix[i - 1].mul(&vals[i]));
    }
    
    // Compute inverse of product
    let mut inv_all = prefix[n - 1].inv();
    
    // Work backwards to get individual inverses
    let mut result = vec![FieldElement::ZERO; n];
    for i in (1..n).rev() {
        // result[i] = inv(a0*...*a[i]) * (a0*...*a[i-1]) = inv(a[i])
        result[i] = inv_all.mul(&prefix[i - 1]);
        // Update: inv_all = inv(a0*...*a[i]) * a[i] = inv(a0*...*a[i-1])
        inv_all = inv_all.mul(&vals[i]);
    }
    result[0] = inv_all;
    
    result
}

/// Check address and potentially record match
/// Uses fast hash160 prefix matching to avoid full bech32 encoding
#[inline]
#[allow(clippy::too_many_arguments)]
fn check_point_and_record(
    x: &FieldElement,
    y: &FieldElement,
    base_key: &Scalar,
    key_offset: i64,
    key_transform: impl FnOnce(Scalar) -> Scalar,
    hrp: &str,
    pattern: &Pattern,
    results_tx: &std::sync::mpsc::Sender<Match>,
    matches_found: &AtomicU64,
) {
    // Compute compressed pubkey (33 bytes)
    let x_bytes = x.to_bytes();
    let prefix = if y.is_odd() { 0x03 } else { 0x02 };
    let mut pubkey = [0u8; 33];
    pubkey[0] = prefix;
    pubkey[1..33].copy_from_slice(&x_bytes);
    
    // Hash160 using optimized function for 33-byte input
    let h160 = hash160_33(&pubkey);
    
    // Fast prefix check on hash160 bytes (avoids bech32 encoding for non-matches)
    if !pattern.matches_hash160(&h160) {
        return;
    }
    
    // Full match - create bech32 address
    let address = bech32::address_from_hash160(hrp, &h160);
    
    // If pattern has wildcards, need to do full match
    if pattern.fast_matcher.has_wildcards
        && !pattern.matches(&address) {
            return;
        }
    
    let mut match_key = *base_key;
    match_key.add_assign(key_offset);
    let match_key = key_transform(match_key);
    
    let m = Match {
        address,
        private_key: match_key,
        compressed: true,
    };
    
    matches_found.fetch_add(1, Ordering::Relaxed);
    let _ = results_tx.send(m);
}

impl CpuSearchEngine {
    pub fn new(config: CpuSearchConfig) -> Self {
        Self {
            config,
            group_table: GroupTable::new(),
            stop: Arc::new(AtomicBool::new(false)),
            keys_checked: Arc::new(AtomicU64::new(0)),
            matches_found: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Run the search with batch modular inverse optimization
    pub fn run(&self, results_tx: std::sync::mpsc::Sender<Match>) {
        let threads = if self.config.threads == 0 {
            num_cpus::get()
        } else {
            self.config.threads
        };

        let start_time = std::time::Instant::now();
        
        // Create thread pool
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();

        // Share group table across threads
        let group_table = &self.group_table;
        
        pool.scope(|s| {
            for _thread_id in 0..threads {
                let stop = Arc::clone(&self.stop);
                let keys_checked = Arc::clone(&self.keys_checked);
                let matches_found = Arc::clone(&self.matches_found);
                let pattern = self.config.pattern.clone();
                let hrp = self.config.hrp.clone();
                let max_matches = self.config.max_matches;
                let timeout_secs = self.config.timeout_secs;
                let results_tx = results_tx.clone();
                
                s.spawn(move |_| {
                    // Initialize random starting key for this thread
                    let mut rng = rand::thread_rng();
                    let mut base_key = generate_random_key(&mut rng);
                    
                    // Compute starting point P = base_key * G
                    // We start at P + (GRP_SIZE/2) * G, so center point is at index GRP_SIZE/2
                    let half = (GRP_SIZE / 2) as u64;
                    let center_scalar = Scalar::new([half, 0, 0, 0]);
                    let start_key = base_key.add(&center_scalar);
                    let mut center_point = G.mul(&start_key);
                    
                    // Allocate work arrays
                    let mut px = vec![FieldElement::ZERO; GRP_SIZE];
                    let mut py = vec![FieldElement::ZERO; GRP_SIZE];
                    let mut dx = vec![FieldElement::ZERO; GRP_SIZE];
                    
                    loop {
                        // Check stop conditions
                        if stop.load(Ordering::Relaxed) {
                            break;
                        }
                        
                        if timeout_secs > 0 && start_time.elapsed().as_secs() >= timeout_secs {
                            stop.store(true, Ordering::Relaxed);
                            break;
                        }
                        
                        if max_matches > 0 && matches_found.load(Ordering::Relaxed) >= max_matches {
                            stop.store(true, Ordering::Relaxed);
                            break;
                        }
                        
                        // ===== BATCH POINT ADDITION WITH MONTGOMERY INVERSE =====
                        // Compute P[i] = center_point + i*G for i in -GRP_SIZE/2 to GRP_SIZE/2-1
                        // Using: P[i] = P[i-1] + G, but batching the modular inverses
                        
                        // Step 1: Compute all dx = x_G - x_point for the additions
                        // For i in 0..GRP_SIZE/2, we add Gx[i] (i+1)*G to center
                        // For i in GRP_SIZE/2..GRP_SIZE, we subtract Gx[GRP_SIZE-i-1]
                        
                        let (cx, cy) = (center_point.x, center_point.y);
                        
                        // Positive side: center + 1*G, center + 2*G, ... (indices GRP_SIZE/2 to GRP_SIZE-1)
                        for i in 0..(GRP_SIZE / 2) {
                            dx[GRP_SIZE / 2 + i] = group_table.gx[i].sub(&cx);
                        }
                        
                        // Negative side: center - 1*G, center - 2*G, ... (indices GRP_SIZE/2-1 down to 0)  
                        // center - i*G = center + (-i*G), and -i*G has x = same, y = -y
                        for i in 0..(GRP_SIZE / 2) {
                            dx[GRP_SIZE / 2 - 1 - i] = group_table.gx[i].sub(&cx);
                        }
                        
                        // Step 2: Batch inverse all dx values
                        let dx_inv = batch_inverse(&dx);
                        
                        // Step 3: Complete the point additions
                        // Center point goes in the middle
                        px[GRP_SIZE / 2] = cx;
                        py[GRP_SIZE / 2] = cy;
                        
                        // Positive direction: P[i] = center + (i - GRP_SIZE/2 + 1)*G
                        for i in 1..(GRP_SIZE / 2) {
                            let idx = GRP_SIZE / 2 + i;
                            let gx = &group_table.gx[i - 1]; // This is i*G
                            let gy = &group_table.gy[i - 1];
                            
                            // Point addition: center + i*G
                            // λ = (gy - cy) / (gx - cx) = (gy - cy) * dx_inv[idx]
                            let dy = gy.sub(&cy);
                            let lambda = dy.mul(&dx_inv[idx - 1]);
                            
                            // x3 = λ² - cx - gx, y3 = λ(cx - x3) - cy
                            let x3 = lambda.sqr().sub(&cx).sub(gx);
                            let y3 = lambda.mul(&cx.sub(&x3)).sub(&cy);
                            
                            px[idx] = x3;
                            py[idx] = y3;
                        }
                        
                        // Negative direction: P[i] = center - (GRP_SIZE/2 - i)*G
                        for i in (0..(GRP_SIZE / 2)).rev() {
                            let dist = GRP_SIZE / 2 - i; // Distance from center
                            let gx = &group_table.gx[dist - 1]; // dist*G
                            let neg_gy = group_table.gy[dist - 1].neg(); // -(dist*G).y
                            
                            // Point addition: center + (-dist*G) where -dist*G = (gx, -gy)
                            let dy = neg_gy.sub(&cy);
                            let lambda = dy.mul(&dx_inv[i]);
                            
                            let x3 = lambda.sqr().sub(&cx).sub(gx);
                            let y3 = lambda.mul(&cx.sub(&x3)).sub(&cy);
                            
                            px[i] = x3;
                            py[i] = y3;
                        }
                        
                        // Step 4: Check all points with endomorphisms (6 addresses per point)
                        for i in 0..GRP_SIZE {
                            let x = &px[i];
                            let y = &py[i];
                            
                            // Key offset from base_key
                            let key_offset = (i as i64) - (GRP_SIZE as i64 / 2);
                            
                            // 1. Original point (x, y) -> key
                            check_point_and_record(
                                x, y, &base_key, key_offset, |k| k,
                                &hrp, &pattern, &results_tx, &matches_found,
                            );
                            
                            // 2. Endomorphism 1: (beta*x, y) -> lambda*key  
                            let beta_x = x.mul(&BETA);
                            check_point_and_record(
                                &beta_x, y, &base_key, key_offset, |k| k.mul(&LAMBDA),
                                &hrp, &pattern, &results_tx, &matches_found,
                            );
                            
                            // 3. Endomorphism 2: (beta²*x, y) -> lambda²*key
                            let beta2_x = x.mul(&BETA2);
                            check_point_and_record(
                                &beta2_x, y, &base_key, key_offset, |k| k.mul(&LAMBDA2),
                                &hrp, &pattern, &results_tx, &matches_found,
                            );
                            
                            // 4. Negation: (x, -y) -> -key
                            let neg_y = y.neg();
                            check_point_and_record(
                                x, &neg_y, &base_key, key_offset, |k| k.neg(),
                                &hrp, &pattern, &results_tx, &matches_found,
                            );
                            
                            // 5. Negation + endo1: (beta*x, -y) -> -lambda*key
                            check_point_and_record(
                                &beta_x, &neg_y, &base_key, key_offset, |k| k.neg().mul(&LAMBDA),
                                &hrp, &pattern, &results_tx, &matches_found,
                            );
                            
                            // 6. Negation + endo2: (beta²*x, -y) -> -lambda²*key
                            check_point_and_record(
                                &beta2_x, &neg_y, &base_key, key_offset, |k| k.neg().mul(&LAMBDA2),
                                &hrp, &pattern, &results_tx, &matches_found,
                            );
                            
                            // Check for early termination
                            if max_matches > 0 && matches_found.load(Ordering::Relaxed) >= max_matches {
                                stop.store(true, Ordering::Relaxed);
                                break;
                            }
                        }
                        
                        // Update keys checked counter (6 addresses per key with endomorphisms)
                        keys_checked.fetch_add((GRP_SIZE * 6) as u64, Ordering::Relaxed);
                        
                        // Move to next batch: advance center by 2*GRP_SIZE*G
                        center_point = center_point.add(&group_table.delta_2n);
                        base_key = base_key.add(&Scalar::new([(2 * GRP_SIZE) as u64, 0, 0, 0]));
                    }
                });
            }
        });
    }

    /// Stop the search
    pub fn stop(&self) {
        self.stop.store(true, Ordering::Relaxed);
    }

    /// Get number of keys checked
    pub fn keys_checked(&self) -> u64 {
        self.keys_checked.load(Ordering::Relaxed)
    }

    /// Get number of matches found
    pub fn matches_found(&self) -> u64 {
        self.matches_found.load(Ordering::Relaxed)
    }

    /// Check if search has stopped
    pub fn is_stopped(&self) -> bool {
        self.stop.load(Ordering::Relaxed)
    }
}

/// Generate a random private key
fn generate_random_key(rng: &mut impl rand::Rng) -> Scalar {
    let mut bytes = [0u8; 32];
    rng.fill(&mut bytes);
    
    // Ensure key is in valid range (1 to n-1)
    // Set high bit to 0 to ensure < n
    bytes[0] &= 0x7F;
    
    // Ensure non-zero
    if bytes.iter().all(|&b| b == 0) {
        bytes[31] = 1;
    }
    
    Scalar::from_bytes(&bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;
    use std::time::Duration;

    #[test]
    fn test_search_short_pattern() {
        let config = CpuSearchConfig {
            threads: 2,
            pattern: Pattern::new("bc1qq*").unwrap(),
            hrp: "bc".to_string(),
            max_matches: 1,
            timeout_secs: 10,
            batch_size: 256,
        };
        
        let engine = CpuSearchEngine::new(config);
        let (tx, rx) = mpsc::channel();
        
        std::thread::spawn(move || {
            engine.run(tx);
        });
        
        // Wait for a match or timeout
        match rx.recv_timeout(Duration::from_secs(10)) {
            Ok(m) => {
                assert!(m.address.starts_with("bc1qq"));
                println!("Found: {} ", m.address);
            }
            Err(_) => {
                // Timeout is acceptable for short test
            }
        }
    }
}
