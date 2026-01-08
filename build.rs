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
fn build_cuda() {
    use std::path::PathBuf;

    let cuda_path = env::var("CUDA_PATH").unwrap_or_else(|_| {
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.1".to_string()
    });

    let cuda_lib = PathBuf::from(&cuda_path).join("lib\\x64");
    let cuda_include = PathBuf::from(&cuda_path).join("include");

    // Link CUDA libraries
    println!("cargo:rustc-link-search=native={}", cuda_lib.display());
    println!("cargo:rustc-link-lib=cuda");
    println!("cargo:rustc-link-lib=cudart");

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
                let vcvars = PathBuf::from(&vs_path).join("VC\\Auxiliary\\Build\\vcvars64.bat");
                if vcvars.exists() {
                    vcvars_path = Some(vcvars);
                }
            }
        }
    }

    // Compile CUDA kernel (GPUBech32.cu which uses original VanitySearch headers)
    let out_dir = env::var("OUT_DIR").unwrap();
    let obj_file = format!("{}/GPUBech32.o", out_dir);
    let lib_file = format!("{}/GPUBech32.lib", out_dir);

    // Build nvcc command with Visual Studio environment
    let status = if let Some(ref vcvars) = vcvars_path {
        let vcvars_str = vcvars.to_str().expect("Invalid vcvars path");
        eprintln!("Using Visual Studio environment: {}", vcvars_str);

        // Run vcvars64.bat in a separate cmd to set environment, then run nvcc
        // We need to pass a script file or batch commands, not try to chain in cmd /C
        // sm_75: Turing (RTX 20xx)
        // sm_86: Ampere (RTX 30xx)
        // sm_89: Ada Lovelace (RTX 40xx)
        // Note: sm_120 (Blackwell/RTX 50xx) requires CUDA 12.6+, excluded for compatibility
        let bat_script = format!(
            r#"@echo off
call "{}"
if errorlevel 1 exit /b 1
nvcc -c cuda\GPUBech32.cu -o "{}" --generate-code arch=compute_75,code=sm_75 --generate-code arch=compute_86,code=sm_86 --generate-code arch=compute_89,code=sm_89 -I "{}" -I cuda -O3 --compiler-options /MD -allow-unsupported-compiler
"#,
            vcvars_str,
            obj_file,
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
        std::process::Command::new("nvcc")
            .args([
                "-c",
                "cuda/GPUBech32.cu",
                "-o",
                &obj_file,
                "--generate-code",
                "arch=compute_75,code=sm_75",
                "--generate-code",
                "arch=compute_86,code=sm_86",
                "--generate-code",
                "arch=compute_89,code=sm_89",
                "-I",
                cuda_include.to_str().unwrap(),
                "-I",
                "cuda",
                "-O3",
                "--compiler-options",
                "/MD",
            ])
            .status()
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

    println!("cargo:rustc-link-search=native={}", out_dir);
    println!("cargo:rustc-link-lib=static=GPUBech32");
}

#[cfg(not(feature = "cuda"))]
fn build_cuda() {
    // CUDA not enabled, do nothing
}
