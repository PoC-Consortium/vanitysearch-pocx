//! Build script for CUDA integration

use std::env;

fn main() {
    // Check if CUDA support is enabled
    if env::var("CARGO_FEATURE_CUDA").is_ok() || cfg!(feature = "cuda") {
        build_cuda();
    }

    println!("cargo:rerun-if-changed=cuda/GPUBech32.cu");
    println!("cargo:rerun-if-changed=cuda/GPUBech32.h");
    println!("cargo:rerun-if-changed=cuda/GPUMath.h");
    println!("cargo:rerun-if-changed=cuda/GPUHash.h");
    println!("cargo:rerun-if-changed=cuda/GPUGroup.h");
    println!("cargo:rerun-if-changed=opencl/bech32_kernel.cl");
    println!("cargo:rerun-if-changed=opencl/group.cl");
    println!("cargo:rerun-if-changed=opencl/hash.cl");
    println!("cargo:rerun-if-changed=opencl/math.cl");
    println!("cargo:rerun-if-changed=build.rs");
}

#[cfg(feature = "cuda")]
fn detect_cuda_version(cuda_path: &str) -> u32 {
    use std::path::PathBuf;

    // Try to get version from nvcc --version
    #[cfg(target_os = "windows")]
    let nvcc = PathBuf::from(cuda_path).join("bin").join("nvcc.exe");
    #[cfg(not(target_os = "windows"))]
    let nvcc = PathBuf::from(cuda_path).join("bin").join("nvcc");
    
    if let Ok(output) = std::process::Command::new(nvcc).arg("--version").output() {
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            // Parse "release 12.6" or "release 13.1"
            if let Some(idx) = stdout.find("release ") {
                let version_str = &stdout[idx + 8..].split_whitespace().next().unwrap_or("0.0");
                if let Some((major, minor)) = version_str.split_once('.') {
                    if let (Ok(maj), Ok(min)) = (major.parse::<u32>(), minor.parse::<u32>()) {
                        return maj * 1000 + min * 10;
                    }
                }
            }
        }
    }

    // Fallback: try to infer from path (v12.6, v13.1, etc.)
    let version_search = if cfg!(target_os = "windows") { "\\v" } else { "/v" };
    if let Some(idx) = cuda_path.rfind(version_search) {
        let version_str = &cuda_path[idx + 2..];
        if let Some((major, minor)) = version_str.split_once('.') {
            if let (Ok(maj), Ok(min)) = (major.parse::<u32>(), minor.parse::<u32>()) {
                return maj * 1000 + min * 10;
            }
        }
    }

    // Default to 12.6
    12_060
}

#[cfg(feature = "cuda")]
fn build_cuda() {
    use std::path::PathBuf;

    let cuda_path = env::var("CUDA_PATH").or_else(|_| env::var("CUDA_HOME")).unwrap_or_else(|_| {
        if cfg!(target_os = "windows") {
            "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.6".to_string()
        } else {
            "/usr/local/cuda".to_string()
        }
    });

    let cuda_lib = if cfg!(target_os = "windows") {
        PathBuf::from(&cuda_path).join("lib").join("x64")
    } else {
        PathBuf::from(&cuda_path).join("lib64")
    };
    
    let cuda_include = PathBuf::from(&cuda_path).join("include");

    // Link CUDA libraries
    println!("cargo:rustc-link-search=native={}", cuda_lib.display());
    println!("cargo:rustc-link-lib=cuda");
    println!("cargo:rustc-link-lib=cudart");

    // Detect CUDA version
    let cuda_version = detect_cuda_version(&cuda_path);
    let supports_sm120 = cuda_version >= 12_006; // CUDA 12.6+

    eprintln!(
        "Detected CUDA version: {}.{}",
        cuda_version / 1000,
        (cuda_version % 1000) / 10
    );
    if supports_sm120 {
        eprintln!("Building with sm_120 support (Blackwell/RTX 50xx)");
    } else {
        eprintln!("sm_120 requires CUDA 12.6+, building without it");
    }

    // Compile CUDA kernel (GPUBech32.cu which uses original VanitySearch headers)
    let out_dir = env::var("OUT_DIR").unwrap();
    
    #[cfg(target_os = "windows")]
    {
        build_cuda_windows(&cuda_path, &cuda_include, &out_dir, cuda_version, supports_sm120);
    }
    
    #[cfg(not(target_os = "windows"))]
    {
        build_cuda_linux(&cuda_path, &cuda_include, &out_dir, cuda_version, supports_sm120);
    }

    println!("cargo:rustc-link-search=native={}", out_dir);
    println!("cargo:rustc-link-lib=static=GPUBech32");
}

#[cfg(all(feature = "cuda", target_os = "windows"))]
fn build_cuda_windows(cuda_path: &str, cuda_include: &std::path::Path, out_dir: &str, cuda_version: u32, supports_sm120: bool) {
    use std::path::PathBuf;
    
    let obj_file = format!("{}/GPUBech32.o", out_dir);
    let lib_file = format!("{}/GPUBech32.lib", out_dir);

    // Try to find Visual Studio using vswhere
    let vswhere_output = std::process::Command::new(
        "C:\\Program Files (x86)\\Microsoft Visual Studio\\Installer\\vswhere.exe",
    )
    .args([
        "-latest",
        "-products",
        "*",
        "-requires",
        "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
        "-property",
        "installationPath",
    ])
    .output();

    let mut vcvars_path = None;
    if let Ok(output) = vswhere_output {
        if output.status.success() {
            let vs_path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !vs_path.is_empty() {
                let vcvars = PathBuf::from(&vs_path).join("VC").join("Auxiliary").join("Build").join("vcvars64.bat");
                if vcvars.exists() {
                    vcvars_path = Some(vcvars);
                }
            }
        }
    }

    // Build architecture flags
    let mut arch_flags = vec![
        "--generate-code".to_string(),
        "arch=compute_75,code=sm_75".to_string(), // Turing (RTX 20xx)
        "--generate-code".to_string(),
        "arch=compute_86,code=sm_86".to_string(), // Ampere (RTX 30xx)
        "--generate-code".to_string(),
        "arch=compute_89,code=sm_89".to_string(), // Ada Lovelace (RTX 40xx)
    ];

    if supports_sm120 {
        arch_flags.push("--generate-code".to_string());
        arch_flags.push("arch=compute_120,code=sm_120".to_string()); // Blackwell (RTX 50xx)
    }

    let arch_flags_str = arch_flags.join(" ");

    // Build nvcc command with Visual Studio environment
    let status = if let Some(ref vcvars) = vcvars_path {
        let vcvars_str = vcvars.to_str().expect("Invalid vcvars path");
        eprintln!("Using Visual Studio environment: {}", vcvars_str);

        let bat_script = format!(
            r#"@echo off
call "{}"
if errorlevel 1 exit /b 1
nvcc -c cuda\GPUBech32.cu -o "{}" {} -I "{}" -I cuda -O3 --compiler-options /MD -allow-unsupported-compiler
"#,
            vcvars_str,
            obj_file,
            arch_flags_str,
            cuda_include.to_str().unwrap()
        );

        let bat_path = format!("{}/build_cuda.bat", out_dir);
        std::fs::write(&bat_path, bat_script).expect("Failed to write build script");

        std::process::Command::new("cmd")
            .args(["/C", &bat_path])
            .stdout(std::process::Stdio::inherit())
            .stderr(std::process::Stdio::inherit())
            .status()
    } else {
        eprintln!("Warning: Could not find Visual Studio. Trying direct nvcc...");
        let mut nvcc_args = vec![
            "-c".to_string(),
            "cuda/GPUBech32.cu".to_string(),
            "-o".to_string(),
            obj_file.clone(),
        ];
        nvcc_args.extend(arch_flags);
        nvcc_args.extend(vec![
            "-I".to_string(),
            cuda_include.to_str().unwrap().to_string(),
            "-I".to_string(),
            "cuda".to_string(),
            "-O3".to_string(),
            "--compiler-options".to_string(),
            "/MD".to_string(),
        ]);

        std::process::Command::new("nvcc").args(&nvcc_args).status()
    };

    match status {
        Ok(s) if s.success() => {
            eprintln!("CUDA kernel compiled successfully");
        }
        Ok(s) => {
            eprintln!("CUDA compilation failed with exit code: {:?}", s.code());
            eprintln!("Make sure:");
            eprintln!("  1. Visual Studio with C++ tools is installed");
            eprintln!("  2. CUDA Toolkit is installed at: {}", cuda_path);
            eprintln!("  3. nvcc is in PATH or use vcvars64.bat");
            panic!("CUDA compilation failed");
        }
        Err(e) => {
            eprintln!("Failed to execute build command: {}", e);
            panic!("Failed to execute nvcc");
        }
    }

    // Create static library
    let lib_status = if let Some(ref vcvars) = vcvars_path {
        let vcvars_str = vcvars.to_str().expect("Invalid vcvars path");

        let bat_script = format!(
            r#"@echo off
call "{}"
if errorlevel 1 exit /b 1
lib /OUT:"{}" "{}"
"#,
            vcvars_str, lib_file, obj_file
        );

        let bat_path = format!("{}/build_lib.bat", out_dir);
        std::fs::write(&bat_path, bat_script).expect("Failed to write lib build script");

        std::process::Command::new("cmd")
            .args(["/C", &bat_path])
            .status()
    } else {
        std::process::Command::new("lib")
            .args([&format!("/OUT:{}", lib_file), &obj_file])
            .status()
    };

    match lib_status {
        Ok(s) if s.success() => {
            eprintln!("CUDA library created successfully");
        }
        _ => {
            eprintln!("Failed to create static library");
            panic!("Failed to create static library");
        }
    }
}

#[cfg(all(feature = "cuda", not(target_os = "windows")))]
fn build_cuda_linux(_cuda_path: &str, cuda_include: &std::path::Path, out_dir: &str, _cuda_version: u32, supports_sm120: bool) {
    let obj_file = format!("{}/GPUBech32.o", out_dir);
    let lib_file = format!("{}/libGPUBech32.a", out_dir);

    // Build architecture flags
    let mut arch_flags = vec![
        "--generate-code".to_string(),
        "arch=compute_75,code=sm_75".to_string(), // Turing (RTX 20xx)
        "--generate-code".to_string(),
        "arch=compute_86,code=sm_86".to_string(), // Ampere (RTX 30xx)
        "--generate-code".to_string(),
        "arch=compute_89,code=sm_89".to_string(), // Ada Lovelace (RTX 40xx)
    ];

    if supports_sm120 {
        arch_flags.push("--generate-code".to_string());
        arch_flags.push("arch=compute_120,code=sm_120".to_string()); // Blackwell (RTX 50xx)
    }

    // Compile CUDA kernel
    let mut nvcc_args = vec![
        "-c",
        "cuda/GPUBech32.cu",
        "-o",
        &obj_file,
    ];
    
    let arch_flags_refs: Vec<&str> = arch_flags.iter().map(|s| s.as_str()).collect();
    nvcc_args.extend(arch_flags_refs);
    
    nvcc_args.extend(&[
        "-I",
        cuda_include.to_str().unwrap(),
        "-I",
        "cuda",
        "-O3",
        "--compiler-options",
        "-fPIC",
    ]);

    eprintln!("Compiling CUDA kernel with nvcc...");
    let status = std::process::Command::new("nvcc")
        .args(&nvcc_args)
        .stdout(std::process::Stdio::inherit())
        .stderr(std::process::Stdio::inherit())
        .status();

    match status {
        Ok(s) if s.success() => {
            eprintln!("CUDA kernel compiled successfully");
        }
        Ok(s) => {
            eprintln!("CUDA compilation failed with exit code: {:?}", s.code());
            eprintln!("Make sure:");
            eprintln!("  1. CUDA Toolkit is installed");
            eprintln!("  2. nvcc is in PATH");
            eprintln!("  3. g++ is installed");
            panic!("CUDA compilation failed");
        }
        Err(e) => {
            eprintln!("Failed to execute nvcc: {}", e);
            panic!("Failed to execute nvcc");
        }
    }

    // Create static library using ar
    eprintln!("Creating static library...");
    let lib_status = std::process::Command::new("ar")
        .args(&["rcs", &lib_file, &obj_file])
        .status();

    match lib_status {
        Ok(s) if s.success() => {
            eprintln!("CUDA library created successfully");
        }
        _ => {
            eprintln!("Failed to create static library");
            panic!("Failed to create static library");
        }
    }
    
    // Tell Rust to link against C++ standard library
    println!("cargo:rustc-link-lib=stdc++");
}

#[cfg(not(feature = "cuda"))]
fn build_cuda() {
    // CUDA not enabled, do nothing
}
