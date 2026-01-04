mod config;
mod crypto;
mod gpu;
mod result;
mod search;
mod output;

use clap::Parser;
use config::Config;
use search::VanitySearch;

fn main() -> anyhow::Result<()> {
    let config = Config::parse();
    
    if config.list_gpus {
        println!("Available GPUs:");
        for gpu in gpu::detect::list_gpus() {
            println!("  {:?}", gpu);
        }
        return Ok(());
    }
    
    // Validate prefixes
    for prefix in &config.prefix {
        if !prefix.starts_with("pocx1q") && !prefix.starts_with("tpocx1q") {
            anyhow::bail!("Prefix must start with 'pocx1q' or 'tpocx1q'");
        }
    }
    
    let mut search = VanitySearch::new(config)?;
    let results = search.run()?;
    
    println!("\nFound {} addresses", results.len());
    
    Ok(())
}
