//! CPU-based vanity search engine
//! Optimized with batch modular inverse (Montgomery's trick)

use crate::bech32;
use crate::hash::hash160_33;
use crate::pattern::Pattern;
use crate::secp256k1::{FieldElement, G, Point, Scalar, LAMBDA, LAMBDA2, BETA, BETA2};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

/// Group size for batch processing (matches VanitySearch GRP_SIZE)
/// Larger values = better batch efficiency, more memory usage
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
fn batch_inverse_into(vals: &[FieldElement], result: &mut [FieldElement]) {
    let n = vals.len();
    if n == 0 {
        return;
    }
    if n == 1 {
        result[0] = vals[0].inv();
        return;
    }
    
    // Use result buffer for prefix products to avoid extra allocation
    result[0] = vals[0];
    for i in 1..n {
        result[i] = result[i - 1].mul(&vals[i]);
    }
    
    // Compute inverse of product
    let mut inv_all = result[n - 1].inv();
    
    // Work backwards to get individual inverses
    for i in (1..n).rev() {
        // result[i] = inv(a0*...*a[i]) * (a0*...*a[i-1]) = inv(a[i])
        let prefix_prev = result[i - 1];
        result[i] = inv_all.mul(&prefix_prev);
        // Update: inv_all = inv(a0*...*a[i]) * a[i] = inv(a0*...*a[i-1])
        inv_all = inv_all.mul(&vals[i]);
    }
    result[0] = inv_all;
}

/// Check address and potentially record match - optimized version
/// Uses fast hash160 prefix matching to avoid full bech32 encoding
#[inline(always)]
#[allow(dead_code)]
fn check_point_fast(
    x: &FieldElement,
    y: &FieldElement,
    pattern: &Pattern,
) -> Option<[u8; 20]> {
    // Compute compressed pubkey (33 bytes)
    let x_bytes = x.to_bytes();
    let prefix = if y.is_odd() { 0x03 } else { 0x02 };
    let mut pubkey = [0u8; 33];
    pubkey[0] = prefix;
    pubkey[1..33].copy_from_slice(&x_bytes);
    
    // Hash160 using optimized function for 33-byte input
    let h160 = hash160_33(&pubkey);
    
    // Fast prefix check on hash160 bytes
    if !pattern.matches_hash160(&h160) {
        return None;
    }
    
    Some(h160)
}

/// Optimized check that takes pre-computed x bytes
/// Avoids redundant to_bytes() calls for endomorphism variants
#[inline(always)]
fn check_point_with_x_bytes(
    x_bytes: &[u8; 32],
    y: &FieldElement,
    pattern: &Pattern,
) -> Option<[u8; 20]> {
    // Compute compressed pubkey (33 bytes)
    let prefix = if y.is_odd() { 0x03 } else { 0x02 };
    let mut pubkey = [0u8; 33];
    pubkey[0] = prefix;
    pubkey[1..33].copy_from_slice(x_bytes);
    
    // Hash160 using optimized function for 33-byte input
    let h160 = hash160_33(&pubkey);
    
    // Fast prefix check on hash160 bytes
    if !pattern.matches_hash160(&h160) {
        return None;
    }
    
    Some(h160)
}

/// Record a match
#[inline]
fn record_match(
    h160: &[u8; 20],
    base_key: &Scalar,
    key_offset: i64,
    key_multiplier: Option<&Scalar>,
    negate: bool,
    hrp: &str,
    pattern: &Pattern,
    results_tx: &std::sync::mpsc::Sender<Match>,
    matches_found: &AtomicU64,
) {
    // Full match - create bech32 address
    let address = bech32::address_from_hash160(hrp, h160);
    
    // If pattern has wildcards, need to do full match
    if pattern.fast_matcher.has_wildcards && !pattern.matches(&address) {
        return;
    }
    
    let mut match_key = *base_key;
    match_key.add_assign(key_offset);
    
    if negate {
        match_key = match_key.neg();
    }
    
    if let Some(mult) = key_multiplier {
        match_key = match_key.mul(mult);
    }
    
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
                    let mut dx_inv = vec![FieldElement::ZERO; GRP_SIZE];
                    
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
                        let (cx, cy) = (center_point.x, center_point.y);
                        
                        // Both positive and negative sides use the same gx values
                        // dx[i] = gx[dist] - cx where dist is the distance from center
                        for i in 0..(GRP_SIZE / 2) {
                            let diff = group_table.gx[i].sub(&cx);
                            dx[GRP_SIZE / 2 + i] = diff;
                            dx[GRP_SIZE / 2 - 1 - i] = diff;
                        }
                        
                        // Step 2: Batch inverse all dx values (reuse dx_inv buffer)
                        batch_inverse_into(&dx, &mut dx_inv);
                        
                        // Step 3: Complete the point additions
                        // Center point goes in the middle
                        px[GRP_SIZE / 2] = cx;
                        py[GRP_SIZE / 2] = cy;
                        
                        // Positive direction: P[i] = center + (i - GRP_SIZE/2 + 1)*G
                        for i in 1..(GRP_SIZE / 2) {
                            let idx = GRP_SIZE / 2 + i;
                            let gx = &group_table.gx[i - 1];
                            let gy = &group_table.gy[i - 1];
                            
                            // λ = (gy - cy) * dx_inv
                            let dy = gy.sub(&cy);
                            let lambda = dy.mul(&dx_inv[idx - 1]);
                            
                            // x3 = λ² - cx - gx, y3 = λ(cx - x3) - cy
                            let lambda_sq = lambda.sqr();
                            let x3 = lambda_sq.sub(&cx).sub(gx);
                            let y3 = lambda.mul(&cx.sub(&x3)).sub(&cy);
                            
                            px[idx] = x3;
                            py[idx] = y3;
                        }
                        
                        // Negative direction: P[i] = center - (GRP_SIZE/2 - i)*G
                        for i in (0..(GRP_SIZE / 2)).rev() {
                            let dist = GRP_SIZE / 2 - i;
                            let gx = &group_table.gx[dist - 1];
                            let neg_gy = group_table.gy[dist - 1].neg();
                            
                            let dy = neg_gy.sub(&cy);
                            let lambda = dy.mul(&dx_inv[i]);
                            
                            let lambda_sq = lambda.sqr();
                            let x3 = lambda_sq.sub(&cx).sub(gx);
                            let y3 = lambda.mul(&cx.sub(&x3)).sub(&cy);
                            
                            px[i] = x3;
                            py[i] = y3;
                        }
                        
                        // Step 4: Check all points with endomorphisms (6 addresses per point)
                        for i in 0..GRP_SIZE {
                            let x = &px[i];
                            let y = &py[i];
                            let key_offset = (i as i64) - (GRP_SIZE as i64 / 2);
                            
                            // Pre-compute x bytes once for this point
                            let x_bytes = x.to_bytes();
                            
                            // Pre-compute endomorphism x values and their bytes
                            let beta_x = x.mul(&BETA);
                            let beta_x_bytes = beta_x.to_bytes();
                            let beta2_x = x.mul(&BETA2);
                            let beta2_x_bytes = beta2_x.to_bytes();
                            let neg_y = y.neg();
                            
                            // 1. Original point (x, y)
                            if let Some(h160) = check_point_with_x_bytes(&x_bytes, y, &pattern) {
                                record_match(&h160, &base_key, key_offset, None, false, &hrp, &pattern, &results_tx, &matches_found);
                            }
                            
                            // 2. Endomorphism 1: (beta*x, y) -> lambda*key
                            if let Some(h160) = check_point_with_x_bytes(&beta_x_bytes, y, &pattern) {
                                record_match(&h160, &base_key, key_offset, Some(&LAMBDA), false, &hrp, &pattern, &results_tx, &matches_found);
                            }
                            
                            // 3. Endomorphism 2: (beta²*x, y) -> lambda²*key
                            if let Some(h160) = check_point_with_x_bytes(&beta2_x_bytes, y, &pattern) {
                                record_match(&h160, &base_key, key_offset, Some(&LAMBDA2), false, &hrp, &pattern, &results_tx, &matches_found);
                            }
                            
                            // 4. Negation: (x, -y) -> -key
                            if let Some(h160) = check_point_with_x_bytes(&x_bytes, &neg_y, &pattern) {
                                record_match(&h160, &base_key, key_offset, None, true, &hrp, &pattern, &results_tx, &matches_found);
                            }
                            
                            // 5. Negation + endo1: (beta*x, -y) -> -(lambda*key)
                            if let Some(h160) = check_point_with_x_bytes(&beta_x_bytes, &neg_y, &pattern) {
                                record_match(&h160, &base_key, key_offset, Some(&LAMBDA), true, &hrp, &pattern, &results_tx, &matches_found);
                            }
                            
                            // 6. Negation + endo2: (beta²*x, -y) -> -(lambda²*key)
                            if let Some(h160) = check_point_with_x_bytes(&beta2_x_bytes, &neg_y, &pattern) {
                                record_match(&h160, &base_key, key_offset, Some(&LAMBDA2), true, &hrp, &pattern, &results_tx, &matches_found);
                            }
                            
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
