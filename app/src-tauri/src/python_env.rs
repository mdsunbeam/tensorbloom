//! Python environment manager for TensorBloom.
//! Creates a local venv, installs PyTorch + dependencies, and provides the
//! path to the Python executable for sidecar spawning.

use std::path::{Path, PathBuf};
use std::process::Command;
use serde::Serialize;

#[derive(Serialize, Clone, Debug)]
pub struct PythonEnvStatus {
    pub ready: bool,
    pub python_path: String,
    pub venv_path: String,
    pub error: Option<String>,
    pub stage: String,
}

/// Returns the venv directory path inside the app data directory.
fn venv_dir(app_data: &Path) -> PathBuf {
    app_data.join("python-env")
}

/// Returns the python executable path inside the venv.
fn venv_python(app_data: &Path) -> PathBuf {
    let venv = venv_dir(app_data);
    if cfg!(windows) {
        venv.join("Scripts").join("python.exe")
    } else {
        venv.join("bin").join("python")
    }
}

/// Returns the pip executable path inside the venv.
fn venv_pip(app_data: &Path) -> PathBuf {
    let venv = venv_dir(app_data);
    if cfg!(windows) {
        venv.join("Scripts").join("pip.exe")
    } else {
        venv.join("bin").join("pip")
    }
}

/// Find a system Python 3 installation.
fn find_system_python() -> Option<String> {
    // Try common names in order of preference
    let candidates = if cfg!(windows) {
        vec!["python", "python3", "py -3"]
    } else {
        vec!["python3", "python"]
    };

    for cmd in candidates {
        let parts: Vec<&str> = cmd.split_whitespace().collect();
        let result = Command::new(parts[0])
            .args(&parts[1..])
            .arg("--version")
            .output();

        if let Ok(output) = result {
            if output.status.success() {
                let version = String::from_utf8_lossy(&output.stdout);
                let version_str = version.trim();
                // Verify it's Python 3.9+
                if version_str.starts_with("Python 3.") {
                    let minor: Option<u32> = version_str
                        .strip_prefix("Python 3.")
                        .and_then(|s| s.split('.').next())
                        .and_then(|s| s.parse().ok());
                    if minor.map_or(false, |m| m >= 9) {
                        return Some(cmd.to_string());
                    }
                }
            }
        }
    }
    None
}

/// Check if the venv exists and has torch installed.
pub fn check_env(app_data: &Path) -> PythonEnvStatus {
    let python = venv_python(app_data);

    if !python.exists() {
        return PythonEnvStatus {
            ready: false,
            python_path: String::new(),
            venv_path: venv_dir(app_data).to_string_lossy().to_string(),
            error: None,
            stage: "not_installed".to_string(),
        };
    }

    // Check if torch is importable
    let check = Command::new(&python)
        .arg("-c")
        .arg("import torch; print(torch.__version__)")
        .output();

    match check {
        Ok(output) if output.status.success() => PythonEnvStatus {
            ready: true,
            python_path: python.to_string_lossy().to_string(),
            venv_path: venv_dir(app_data).to_string_lossy().to_string(),
            error: None,
            stage: format!("ready (torch {})", String::from_utf8_lossy(&output.stdout).trim()),
        },
        _ => PythonEnvStatus {
            ready: false,
            python_path: python.to_string_lossy().to_string(),
            venv_path: venv_dir(app_data).to_string_lossy().to_string(),
            error: Some("PyTorch not installed in venv".to_string()),
            stage: "deps_missing".to_string(),
        },
    }
}

/// Set up the Python venv and install dependencies.
/// Returns status updates as it progresses.
pub fn setup_env(app_data: &Path) -> Result<PythonEnvStatus, String> {
    // 1. Find system Python
    let system_python = find_system_python()
        .ok_or("Python 3.9+ not found. Please install Python from python.org")?;

    let venv = venv_dir(app_data);
    let python = venv_python(app_data);

    // 2. Create venv if it doesn't exist
    if !python.exists() {
        let parts: Vec<&str> = system_python.split_whitespace().collect();
        let output = Command::new(parts[0])
            .args(&parts[1..])
            .arg("-m")
            .arg("venv")
            .arg(&venv)
            .output()
            .map_err(|e| format!("Failed to create venv: {}", e))?;

        if !output.status.success() {
            return Err(format!(
                "Failed to create venv: {}",
                String::from_utf8_lossy(&output.stderr)
            ));
        }
    }

    // 3. Upgrade pip
    let pip = venv_pip(app_data);
    let _ = Command::new(&python)
        .args(["-m", "pip", "install", "--upgrade", "pip"])
        .output();

    // 4. Install PyTorch. The CUDA index only hosts PyTorch packages, so torch is
    // installed on its own first; mixing other packages into this command makes
    // pip fail and silently fall back to CPU-only wheels (on Windows).
    let torch_packages = ["torch", "torchvision", "torchaudio"];
    let torch_index = if cfg!(windows) || cfg!(target_os = "linux") {
        // CUDA 12.6 wheels
        Some("--index-url=https://download.pytorch.org/whl/cu126")
    } else {
        // macOS — CPU/MPS wheels from PyPI
        None
    };

    let mut torch_args = vec!["install"];
    torch_args.extend(torch_packages);
    if let Some(index) = torch_index {
        torch_args.push(index);
    }

    let output = Command::new(&pip)
        .args(&torch_args)
        .output()
        .map_err(|e| format!("Failed to install PyTorch: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        // No CUDA build for this platform/Python version — fall back to PyPI wheels
        if torch_index.is_some()
            && (stderr.contains("No matching distribution") || stderr.contains("Could not find"))
        {
            let output2 = Command::new(&pip)
                .arg("install")
                .args(torch_packages)
                .output()
                .map_err(|e| format!("Failed to install CPU packages: {}", e))?;

            if !output2.status.success() {
                return Err(format!(
                    "Failed to install PyTorch: {}",
                    String::from_utf8_lossy(&output2.stderr)
                ));
            }
        } else {
            return Err(format!("Failed to install PyTorch: {}", stderr));
        }
    }

    // 5. Install the remaining dependencies from PyPI
    let output = Command::new(&pip)
        .args(["install", "onnx", "onnxscript", "datasets", "soundfile"])
        .output()
        .map_err(|e| format!("Failed to install packages: {}", e))?;

    if !output.status.success() {
        return Err(format!(
            "Failed to install dependencies: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    Ok(check_env(app_data))
}

/// Get the python executable path for spawning the sidecar.
pub fn get_python_path(app_data: &Path) -> Option<String> {
    let python = venv_python(app_data);
    if python.exists() {
        Some(python.to_string_lossy().to_string())
    } else {
        None
    }
}
