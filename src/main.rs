//! VanitySearch-POCX CLI

use clap::Parser;
use std::sync::mpsc;
use std::time::{Duration, Instant};
use vanitysearch_pocx::{
    CpuSearchConfig, CpuSearchEngine, FormattedMatch, NetworkInfo, Pattern, Stats,
};
#[cfg(feature = "cuda")]
use vanitysearch_pocx::{GpuSearchConfig, GpuSearchEngine};
#[cfg(feature = "opencl")]
use vanitysearch_pocx::{OpenClSearchConfig, OpenClSearchEngine};

#[derive(Parser, Debug)]
#[command(name = "vanitysearch-pocx")]
#[command(author = "VanitySearch-POCX Contributors")]
#[command(version = "0.5.2")]
#[command(about = "Bitcoin bech32 vanity address generator", long_about = None)]
struct Args {
    /// Pattern to search for (e.g., bc1qmadf0*)
    pattern: String,

    /// Number of CPU threads (0 = auto)
    #[arg(short = 't', long, default_value = "0")]
    threads: usize,

    /// Timeout in seconds (0 = infinite)
    #[arg(short = 'T', long, default_value = "0")]
    timeout: u64,

    /// Maximum number of matches to find (0 = infinite)
    #[arg(short = 'm', long, default_value = "0")]
    max_found: u64,

    /// Use GPU mode (auto-detects CUDA, falls back to OpenCL)
    #[arg(short = 'g', long)]
    gpu: bool,

    /// GPU device ID
    #[arg(long, default_value = "0")]
    gpu_id: i32,

    /// Number of GPU threads (default: auto-detect)
    #[arg(long, default_value = "0")]
    gpu_threads: i32,

    /// GPU threads per block (default: 256)
    #[arg(long, default_value = "256")]
    gpu_block_size: i32,

    /// Force OpenCL mode (overrides CUDA detection)
    #[arg(long)]
    opencl: bool,

    /// OpenCL platform ID
    #[arg(long, default_value = "0")]
    opencl_platform: usize,

    /// OpenCL number of work items (default: 65536)
    #[arg(long, default_value = "65536")]
    opencl_threads: usize,

    /// Output format: text, json, csv
    #[arg(short = 'o', long, default_value = "text")]
    output_format: String,

    /// Output file (stdout if not specified)
    #[arg(short = 'f', long)]
    output_file: Option<String>,

    /// Quiet mode (no progress output)
    #[arg(short = 'q', long)]
    quiet: bool,

    /// Verbose mode (print full details for each match immediately)
    #[arg(short = 'v', long)]
    verbose: bool,
}

fn main() {
    let args = Args::parse();

    // Parse pattern
    let pattern = match Pattern::new(&args.pattern) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Error: Invalid pattern: {}", e);
            std::process::exit(1);
        }
    };

    // Fixed networks: Mainnet and Testnet
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

    if !args.quiet {
        eprintln!("VanitySearch-POCX v0.5.2");
        eprintln!("Pattern: {}", args.pattern);
        eprintln!("Difficulty: ~2^{:.1}", pattern.difficulty.log2());
        if args.timeout > 0 {
            eprintln!("Timeout: {}s", args.timeout);
        }
        if args.max_found > 0 {
            eprintln!("Max matches: {}", args.max_found);
        }
        eprintln!();
    }

    // Determine GPU mode: --opencl forces OpenCL, -g tries CUDA first then OpenCL
    if args.opencl || args.gpu {
        // --opencl flag forces OpenCL mode
        if args.opencl {
            run_opencl_search(&args, pattern, networks);
        } else {
            // -g flag: Try CUDA first, fallback to OpenCL if CUDA unavailable
            #[cfg(feature = "cuda")]
            {
                let cuda_available = GpuSearchEngine::device_count() > 0;
                if cuda_available {
                    run_gpu_search(&args, pattern, networks);
                } else {
                    if !args.quiet {
                        eprintln!("CUDA not available, falling back to OpenCL...\n");
                    }
                    run_opencl_search(&args, pattern, networks);
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                run_opencl_search(&args, pattern, networks);
            }
        }
    } else {
        run_cpu_search(&args, pattern, networks);
    }
}

fn run_cpu_search(args: &Args, pattern: Pattern, networks: Vec<NetworkInfo>) {
    // Create search config
    let config = CpuSearchConfig {
        threads: args.threads,
        pattern: pattern.clone(),
        hrp: pattern.hrp.clone(),
        max_matches: args.max_found,
        timeout_secs: args.timeout,
        batch_size: 1024,
    };

    let engine = CpuSearchEngine::new(config);
    let (tx, rx) = mpsc::channel();

    // Start search in background thread
    let engine_handle = {
        let engine = std::sync::Arc::new(engine);
        let engine_clone = std::sync::Arc::clone(&engine);

        std::thread::spawn(move || {
            engine_clone.run(tx);
        });

        engine
    };

    run_main_loop(args, rx, &*engine_handle, &networks);
}

#[allow(dead_code)]
fn run_gpu_search(_args: &Args, _pattern: Pattern, _networks: Vec<NetworkInfo>) {
    #[cfg(not(feature = "cuda"))]
    {
        eprintln!(
            "Error: GPU support requires CUDA feature. Build with: cargo build --features cuda"
        );
        std::process::exit(1);
    }

    #[cfg(feature = "cuda")]
    run_gpu_search_cuda(_args, _pattern, _networks);
}

fn run_opencl_search(_args: &Args, _pattern: Pattern, _networks: Vec<NetworkInfo>) {
    #[cfg(not(feature = "opencl"))]
    {
        eprintln!(
            "Error: OpenCL support requires opencl feature. Build with: cargo build --features opencl"
        );
        std::process::exit(1);
    }

    #[cfg(feature = "opencl")]
    run_opencl_search_impl(_args, _pattern, _networks);
}

#[cfg(feature = "opencl")]
fn run_opencl_search_impl(args: &Args, pattern: Pattern, networks: Vec<NetworkInfo>) {
    // OpenCL mode does NOT support wildcards - check and abort early
    if pattern.fast_matcher.has_wildcards {
        eprintln!("Error: OpenCL mode does not support wildcards in patterns.");
        eprintln!("Pattern '{}' contains wildcards (? or *).", args.pattern);
        eprintln!("Please use CPU mode for wildcard patterns, or use a pattern without wildcards.");
        std::process::exit(1);
    }

    // List available platforms and devices
    let platforms = OpenClSearchEngine::list_platforms();
    if platforms.is_empty() {
        eprintln!("Error: No OpenCL platforms found.");
        std::process::exit(1);
    }

    if !args.quiet {
        eprintln!(
            "OpenCL Mode - {} device(s) found",
            OpenClSearchEngine::list_devices(args.opencl_platform).len()
        );
        eprintln!(
            "  Device {}: {}",
            args.gpu_id,
            OpenClSearchEngine::list_devices(args.opencl_platform)
                .get(args.gpu_id.max(0) as usize)
                .unwrap_or(&"Unknown".to_string())
        );
        eprintln!();
    }

    // Create OpenCL search config
    let config = OpenClSearchConfig {
        platform_id: args.opencl_platform,
        device_id: args.gpu_id.max(0) as usize, // Use --gpu-id instead of separate --opencl-device
        pattern: pattern.clone(),
        hrp: pattern.hrp.clone(),
        max_matches: args.max_found,
        timeout_secs: args.timeout,
        num_threads: if args.opencl_threads == 65536 && args.gpu_threads > 0 {
            args.gpu_threads as usize // Use --gpu-threads if specified
        } else {
            args.opencl_threads
        },
    };

    let engine = match OpenClSearchEngine::new(config) {
        Ok(e) => e,
        Err(err) => {
            eprintln!("Error: Failed to initialize OpenCL: {}", err);
            std::process::exit(1);
        }
    };

    if !args.quiet {
        eprintln!("GPU: {}", engine.device_name());
        eprintln!();
    }

    let (tx, rx) = mpsc::channel();

    let engine_handle = std::sync::Arc::new(engine);
    let engine_clone = std::sync::Arc::clone(&engine_handle);

    let thread_handle = std::thread::spawn(move || {
        engine_clone.run(tx);
    });

    run_main_loop(args, rx, &*engine_handle, &networks);

    // Signal stop and wait for thread to finish cleanly
    engine_handle.stop();
    let _ = thread_handle.join();
}

#[cfg(feature = "cuda")]
fn run_gpu_search_cuda(args: &Args, pattern: Pattern, networks: Vec<NetworkInfo>) {
    // GPU mode does NOT support wildcards - check and abort early
    if pattern.fast_matcher.has_wildcards {
        eprintln!("Error: GPU mode does not support wildcards in patterns.");
        eprintln!("Pattern '{}' contains wildcards (? or *).", args.pattern);
        eprintln!("Please use CPU mode for wildcard patterns, or use a pattern without wildcards.");
        std::process::exit(1);
    }

    // Check GPU availability
    let device_count = GpuSearchEngine::device_count();
    if device_count == 0 {
        eprintln!("Error: No CUDA devices found.");
        std::process::exit(1);
    }

    if !args.quiet {
        eprintln!("CUDA Mode - {} device(s) found", device_count);
        eprintln!(
            "  Device {}: {}",
            args.gpu_id,
            GpuSearchEngine::device_name(args.gpu_id)
        );
        eprintln!();
    }

    let gpu_threads = if args.gpu_threads == 0 {
        // Auto-detect optimal thread count based on SM count
        // VanitySearch uses: num_blocks = multiProcessorCount * 8
        // Default threads_per_block = 128
        let sm_count = GpuSearchEngine::device_sm_count(args.gpu_id);
        let optimal = sm_count * 8 * 128;
        eprintln!("GPU: {}", GpuSearchEngine::device_name(args.gpu_id));
        eprintln!("SM Count: {}", sm_count);
        eprintln!("Threads: {}", optimal);
        eprintln!();
        optimal
    } else {
        args.gpu_threads
    };

    // Create GPU search config
    // num_thread_groups is the number of blocks, threads_per_group is threads per block
    // Total threads = num_thread_groups * threads_per_group
    // For maximum performance: use blocks = total_threads / threads_per_block
    let threads_per_block = if args.gpu_block_size == 0 {
        256
    } else {
        args.gpu_block_size
    };
    let num_blocks = (gpu_threads as i32) / threads_per_block;

    let config = GpuSearchConfig {
        device_id: args.gpu_id,
        pattern: pattern.clone(),
        hrp: pattern.hrp.clone(),
        max_matches: args.max_found,
        timeout_secs: args.timeout,
        num_thread_groups: num_blocks,
        threads_per_group: threads_per_block,
    };

    let engine = match GpuSearchEngine::new(config) {
        Ok(e) => e,
        Err(err) => {
            eprintln!("Error: Failed to initialize GPU: {}", err);
            std::process::exit(1);
        }
    };

    let (tx, rx) = mpsc::channel();

    let engine_handle = std::sync::Arc::new(engine);
    let engine_clone = std::sync::Arc::clone(&engine_handle);

    let thread_handle = std::thread::spawn(move || {
        engine_clone.run(tx);
    });

    run_main_loop_gpu(args, rx, &engine_handle, &networks);

    // Signal stop and wait for thread to finish cleanly
    engine_handle.stop();
    let _ = thread_handle.join();
}

trait SearchEngine {
    fn keys_checked(&self) -> u64;
    fn matches_found(&self) -> u64;
    fn is_stopped(&self) -> bool;
    fn stop(&self);
}

impl SearchEngine for CpuSearchEngine {
    fn keys_checked(&self) -> u64 {
        self.keys_checked()
    }
    fn matches_found(&self) -> u64 {
        self.matches_found()
    }
    fn is_stopped(&self) -> bool {
        self.is_stopped()
    }
    fn stop(&self) {
        self.stop()
    }
}

#[cfg(feature = "cuda")]
impl SearchEngine for GpuSearchEngine {
    fn keys_checked(&self) -> u64 {
        self.keys_checked()
    }
    fn matches_found(&self) -> u64 {
        self.matches_found()
    }
    fn is_stopped(&self) -> bool {
        self.is_stopped()
    }
    fn stop(&self) {
        self.stop()
    }
}

#[cfg(feature = "opencl")]
impl SearchEngine for OpenClSearchEngine {
    fn keys_checked(&self) -> u64 {
        self.keys_checked()
    }
    fn matches_found(&self) -> u64 {
        self.matches_found()
    }
    fn is_stopped(&self) -> bool {
        self.is_stopped()
    }
    fn stop(&self) {
        self.stop()
    }
}

fn run_main_loop<E: SearchEngine>(
    args: &Args,
    rx: mpsc::Receiver<vanitysearch_pocx::search::Match>,
    engine: &E,
    networks: &[NetworkInfo],
) {
    let start_time = Instant::now();
    let mut last_stats_time = Instant::now();
    let mut output_file: Option<std::fs::File> = args
        .output_file
        .as_ref()
        .map(|path| std::fs::File::create(path).expect("Failed to create output file"));

    // Collect all matches for final output
    let mut all_matches: Vec<vanitysearch_pocx::search::Match> = Vec::new();

    // Main loop
    loop {
        // Check for results
        match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(m) => {
                if !args.quiet {
                    // Clear current line
                    eprint!("\r{:80}\r", "");

                    if args.verbose {
                        // Print full details immediately
                        let formatted = FormattedMatch::from_match(&m, networks);
                        eprintln!("{}", formatted.to_text_colored(true));
                        eprintln!();
                    } else {
                        // Print just the address during search
                        eprintln!("  Found: {}", m.address);
                    }
                }
                all_matches.push(m);
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }

        // Print stats periodically
        if !args.quiet && last_stats_time.elapsed() >= Duration::from_secs(1) {
            let elapsed = start_time.elapsed().as_secs_f64();
            let keys_checked = engine.keys_checked();
            let matches_found = engine.matches_found();
            let keys_per_second = keys_checked as f64 / elapsed;

            let stats = Stats {
                keys_per_second,
                total_keys: keys_checked,
                matches_found,
                elapsed_secs: elapsed,
            };

            eprint!("\r{}", stats.format());
            last_stats_time = Instant::now();
        }

        // Check stop conditions
        if engine.is_stopped() {
            break;
        }

        if args.timeout > 0 && start_time.elapsed().as_secs() >= args.timeout {
            engine.stop();
            break;
        }
    }

    // Wait a tiny bit for final messages
    std::thread::sleep(Duration::from_millis(50));

    // Final stats
    let elapsed = start_time.elapsed().as_secs_f64();
    let keys_checked = engine.keys_checked();
    let matches_found = engine.matches_found();
    let keys_per_second = keys_checked as f64 / elapsed;

    if !args.quiet {
        // Clear the progress line and print final stats
        eprint!("\r{:80}\r", ""); // Clear progress line
        eprintln!();
        eprintln!("Search completed.");
        eprintln!("Total time: {:.2}s", elapsed);
        eprintln!("Keys checked: {}", keys_checked);
        eprintln!("Average speed: {:.2} Mkey/s", keys_per_second / 1_000_000.0);
        eprintln!("Matches found: {}", matches_found);
    }

    // Output all matches with full details
    if !all_matches.is_empty() {
        // Only print results section if not in verbose mode (verbose already printed details)
        if !args.verbose {
            if !args.quiet {
                eprintln!();
                eprintln!("{}", "=".repeat(85));
                eprintln!("||{}RESULTS{}||", " ".repeat(37), " ".repeat(37));
                eprintln!("{}", "=".repeat(85));
                eprintln!();
            }

            for m in &all_matches {
                let formatted = FormattedMatch::from_match(m, networks);

                let output = match args.output_format.as_str() {
                    "json" => formatted.to_json(),
                    "csv" => formatted.to_csv(),
                    _ => formatted.to_text_colored(true), // Use colored output for text
                };

                if let Some(ref mut file) = output_file {
                    use std::io::Write;
                    // Write without colors to file
                    let file_output = match args.output_format.as_str() {
                        "json" => formatted.to_json(),
                        "csv" => formatted.to_csv(),
                        _ => formatted.to_text(), // No colors for file output
                    };
                    writeln!(file, "{}", file_output).expect("Failed to write to output file");
                    writeln!(file).ok();
                } else {
                    // Use eprintln for consistent output stream
                    eprintln!("{}", output);
                    eprintln!();
                }
            }
        } else if let Some(ref mut file) = output_file {
            // In verbose mode, still write to file if specified
            use std::io::Write;
            for m in &all_matches {
                let formatted = FormattedMatch::from_match(m, networks);
                let file_output = match args.output_format.as_str() {
                    "json" => formatted.to_json(),
                    "csv" => formatted.to_csv(),
                    _ => formatted.to_text(), // No colors for file output
                };
                writeln!(file, "{}", file_output).expect("Failed to write to output file");
                writeln!(file).ok();
            }
        }
    }
}

// Separate GPU main loop due to ownership issues
#[cfg(feature = "cuda")]
fn run_main_loop_gpu(
    args: &Args,
    rx: mpsc::Receiver<vanitysearch_pocx::search::Match>,
    engine: &GpuSearchEngine,
    networks: &[NetworkInfo],
) {
    run_main_loop(args, rx, engine, networks);
}
