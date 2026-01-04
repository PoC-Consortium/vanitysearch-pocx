use crate::result::SearchResult;
use std::fs::OpenOptions;
use std::io::Write;

#[allow(dead_code)]
pub fn save_result(result: &SearchResult, output_file: Option<&str>) -> std::io::Result<()> {
    let output = format!("{}\n", result);
    
    if let Some(path) = output_file {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        file.write_all(output.as_bytes())?;
    }
    
    Ok(())
}
