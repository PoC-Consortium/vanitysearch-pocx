use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "pocx-vanity")]
#[command(about = "PoCX Bech32 Vanity Address Generator")]
pub struct Config {
    /// Prefix to search (e.g., "pocx1qtest")
    #[arg(short, long, required = true)]
    pub prefix: Vec<String>,

    /// Force specific backend: "cuda" or "opencl"
    #[arg(long)]
    pub backend: Option<String>,

    /// GPU device ID
    #[arg(short, long, default_value = "0")]
    pub gpu: u32,

    /// Stop after finding N addresses
    #[arg(short, long, default_value = "1")]
    pub max_found: u32,

    /// Output file (stdout if not specified)
    #[arg(short, long)]
    pub output: Option<String>,

    /// Seed for deterministic key generation
    #[arg(long)]
    pub seed: Option<String>,

    /// Grid size (auto if not specified)
    #[arg(long)]
    pub grid_size: Option<u32>,

    /// List available GPUs
    #[arg(long)]
    pub list_gpus: bool,
}
