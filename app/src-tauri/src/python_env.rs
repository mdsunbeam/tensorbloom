//! Python environment manager for TensorBloom.
//! Creates a local venv, installs PyTorch + dependencies, and provides the
//! path to the Python executable for sidecar spawning.

use std::collections::VecDeque;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use serde::Serialize;

#[cfg(windows)]
use std::os::windows::process::CommandExt;

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

/// What `setup_env` reports while it works.
#[derive(Serialize, Clone, Debug)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SetupUpdate {
    /// A new step started
    Step { step: u32, total: u32, label: String },
    /// Bytes downloaded so far of the file pip is fetching
    Download { done: u64, total: u64 },
    /// A line of output from pip or Python
    Log { line: String },
}

const SETUP_STEPS: u32 = 6;

/// Parses pip's `--progress-bar=raw` lines, which look like "Progress 1024 of 2048".
fn parse_pip_progress(line: &str) -> Option<(u64, u64)> {
    let rest = line.trim().strip_prefix("Progress ")?;
    let (done, total) = rest.split_once(" of ")?;
    Some((done.trim().parse().ok()?, total.trim().parse().ok()?))
}

/// Runs a command, passing each line it prints to `report` as it arrives.
/// Returns whether it succeeded and the last lines it wrote to stderr.
fn run_streamed(cmd: &mut Command, report: &(dyn Fn(SetupUpdate) + Sync)) -> Result<(bool, String), String> {
    #[cfg(windows)]
    cmd.creation_flags(0x08000000); // CREATE_NO_WINDOW
    let mut child = cmd
        .env("PYTHONUNBUFFERED", "1")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| e.to_string())?;
    let stdout = child.stdout.take().ok_or("No stdout")?;
    let stderr = child.stderr.take().ok_or("No stderr")?;

    let stderr_tail = std::thread::scope(|scope| {
        let err_reader = scope.spawn(|| {
            let mut tail: VecDeque<String> = VecDeque::new();
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                report(SetupUpdate::Log { line: line.clone() });
                tail.push_back(line);
                if tail.len() > 20 {
                    tail.pop_front();
                }
            }
            Vec::from(tail).join("\n")
        });
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            match parse_pip_progress(&line) {
                Some((done, total)) => report(SetupUpdate::Download { done, total }),
                None => report(SetupUpdate::Log { line }),
            }
        }
        err_reader.join().unwrap_or_default()
    });

    let status = child.wait().map_err(|e| e.to_string())?;
    Ok((status.success(), stderr_tail))
}

/// `pip install` with byte-level download progress. pip older than 24.1 has no raw
/// progress bar, so it's tried again without one if pip rejects the option.
fn pip_install(python: &Path, args: &[&str], report: &(dyn Fn(SetupUpdate) + Sync)) -> Result<(bool, String), String> {
    let run = |progress: bool| {
        let mut cmd = Command::new(python);
        cmd.args(["-m", "pip", "install", "--disable-pip-version-check"]);
        if progress {
            cmd.arg("--progress-bar=raw");
        }
        cmd.args(args);
        run_streamed(&mut cmd, report)
    };
    let (ok, stderr) = run(true)?;
    if !ok && stderr.contains("invalid choice") {
        return run(false);
    }
    Ok((ok, stderr))
}

/// Set up the Python venv and install dependencies, reporting each step and
/// pip's output through `report` as it goes.
pub fn setup_env(app_data: &Path, report: &(dyn Fn(SetupUpdate) + Sync)) -> Result<PythonEnvStatus, String> {
    let step = |n: u32, label: &str| report(SetupUpdate::Step { step: n, total: SETUP_STEPS, label: label.to_string() });

    // 1. Find system Python
    step(1, "Finding Python");
    let system_python = find_system_python()
        .ok_or("TensorBloom needs Python 3.9 or newer and couldn't find it. Install it from python.org, then try again.")?;
    report(SetupUpdate::Log { line: format!("Using {}", system_python) });

    let venv = venv_dir(app_data);
    let python = venv_python(app_data);

    // 2. Create venv if it doesn't exist
    step(2, "Creating the Python environment");
    if !python.exists() {
        let parts: Vec<&str> = system_python.split_whitespace().collect();
        let (ok, stderr) = run_streamed(
            Command::new(parts[0]).args(&parts[1..]).arg("-m").arg("venv").arg(&venv),
            report,
        )
        .map_err(|e| format!("Failed to create venv: {}", e))?;

        if !ok {
            return Err(format!("Failed to create venv: {}", stderr));
        }
    }

    // 3. Upgrade pip (a newer pip also reports download progress)
    step(3, "Updating pip");
    let _ = pip_install(&python, &["--upgrade", "pip"], report);

    // 4. Install PyTorch. The CUDA index only hosts PyTorch packages, so torch is
    // installed on its own first; mixing other packages into this command makes
    // pip fail and silently fall back to CPU-only wheels (on Windows).
    step(4, "Downloading PyTorch");
    let torch_packages = ["torch", "torchvision", "torchaudio"];
    let torch_index = if cfg!(windows) || cfg!(target_os = "linux") {
        // CUDA 12.6 wheels
        Some("--index-url=https://download.pytorch.org/whl/cu126")
    } else {
        // macOS — CPU/MPS wheels from PyPI
        None
    };

    let mut torch_args: Vec<&str> = torch_packages.to_vec();
    if let Some(index) = torch_index {
        torch_args.push(index);
    }

    let (ok, stderr) = pip_install(&python, &torch_args, report)
        .map_err(|e| format!("Failed to install PyTorch: {}", e))?;

    if !ok {
        // No CUDA build for this platform/Python version — fall back to PyPI wheels
        if torch_index.is_some()
            && (stderr.contains("No matching distribution") || stderr.contains("Could not find"))
        {
            report(SetupUpdate::Log { line: "No CUDA build of PyTorch for this system, installing the standard one".to_string() });
            let (ok2, stderr2) = pip_install(&python, &torch_packages, report)
                .map_err(|e| format!("Failed to install CPU packages: {}", e))?;

            if !ok2 {
                return Err(format!("Failed to install PyTorch: {}", stderr2));
            }
        } else {
            return Err(format!("Failed to install PyTorch: {}", stderr));
        }
    }

    // 5. Install the remaining dependencies from PyPI
    step(5, "Installing the other packages");
    let (ok, stderr) = pip_install(&python, &["onnx", "onnxscript", "datasets", "soundfile"], report)
        .map_err(|e| format!("Failed to install packages: {}", e))?;

    if !ok {
        return Err(format!("Failed to install dependencies: {}", stderr));
    }

    // 6. Make sure torch imports
    step(6, "Checking the install");
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

#[cfg(test)]
mod tests {
    use super::parse_pip_progress;

    #[test]
    fn reads_pip_raw_progress() {
        assert_eq!(parse_pip_progress("Progress 1024 of 2048"), Some((1024, 2048)));
        assert_eq!(parse_pip_progress("  Progress 0 of 0\r"), Some((0, 0)));
        assert_eq!(parse_pip_progress("Collecting torch"), None);
        assert_eq!(parse_pip_progress("Progress a of b"), None);
    }
}
