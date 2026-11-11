//! Python environment manager for TensorBloom.
//! Uses the bundled uv to download a Python of TensorBloom's own, create a venv
//! with it, and install PyTorch, so nothing has to be installed first. PyTorch
//! starts as the CPU build (small and works everywhere); people with an NVIDIA GPU
//! can switch to the CUDA build afterwards.

use std::collections::{HashMap, VecDeque};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use serde::Serialize;

#[cfg(windows)]
use std::os::windows::process::CommandExt;

/// The Python version TensorBloom installs. PyTorch publishes wheels for it on every platform.
const PYTHON_VERSION: &str = "3.12";

/// The CPU build of PyTorch. macOS uses PyPI instead, whose wheels include MPS support.
const TORCH_CPU_INDEX: &str = "https://download.pytorch.org/whl/cpu";
/// The CUDA build, for the GPU upgrade on Windows and Linux.
const TORCH_CUDA_INDEX: &str = "https://download.pytorch.org/whl/cu126";
const TORCH_PACKAGES: [&str; 3] = ["torch", "torchvision", "torchaudio"];
const OTHER_PACKAGES: [&str; 4] = ["onnx", "onnxscript", "datasets", "soundfile"];

#[derive(Serialize, Clone, Debug)]
pub struct PythonEnvStatus {
    pub ready: bool,
    pub python_path: String,
    pub venv_path: String,
    pub error: Option<String>,
    pub stage: String,
    /// e.g. "2.9.0+cpu", when PyTorch imports
    pub torch_version: Option<String>,
    /// Whether this PyTorch build can use an NVIDIA GPU
    pub torch_cuda: bool,
    /// Whether switching to the CUDA build is offered on this platform
    pub cuda_upgrade_supported: bool,
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

fn cuda_upgrade_supported() -> bool {
    cfg!(windows) || cfg!(target_os = "linux")
}

/// The uv binary bundled next to the app's executable (Tauri's externalBin), or one on PATH
/// for development builds made without `scripts/fetch-uv.mjs`.
fn find_uv() -> Option<PathBuf> {
    let name = if cfg!(windows) { "uv.exe" } else { "uv" };
    if let Some(dir) = std::env::current_exe().ok().and_then(|exe| exe.parent().map(Path::to_path_buf)) {
        let bundled = dir.join(name);
        if bundled.is_file() {
            return Some(bundled);
        }
    }
    let on_path = PathBuf::from(name);
    let works = Command::new(&on_path)
        .arg("--version")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    works.then_some(on_path)
}

/// A uv command that keeps everything inside the app data directory and ignores any
/// uv or Python setup the person has of their own.
fn uv_command(uv: &Path, app_data: &Path) -> Command {
    let mut cmd = Command::new(uv);
    cmd.env("UV_PYTHON_INSTALL_DIR", app_data.join("python"))
        .env("UV_CACHE_DIR", app_data.join("uv-cache"))
        .env("UV_PYTHON_PREFERENCE", "only-managed")
        .env("UV_NO_CONFIG", "1")
        .env("NO_COLOR", "1")
        .env_remove("VIRTUAL_ENV")
        .env_remove("CONDA_PREFIX")
        .env_remove("PYTHONHOME")
        .env_remove("PYTHONPATH");
    cmd
}

/// Check if the venv exists and has torch installed.
pub fn check_env(app_data: &Path) -> PythonEnvStatus {
    let python = venv_python(app_data);
    let venv_path = venv_dir(app_data).to_string_lossy().to_string();
    let not_ready = |python_path: String, error: Option<String>, stage: &str| PythonEnvStatus {
        ready: false,
        python_path,
        venv_path: venv_path.clone(),
        error,
        stage: stage.to_string(),
        torch_version: None,
        torch_cuda: false,
        cuda_upgrade_supported: cuda_upgrade_supported(),
    };

    if !python.exists() {
        return not_ready(String::new(), None, "not_installed");
    }

    let mut check = Command::new(&python);
    check
        .arg("-c")
        .arg("import torch; print(torch.__version__); print(torch.version.cuda or '')")
        .stdin(Stdio::null());
    #[cfg(windows)]
    check.creation_flags(0x08000000); // CREATE_NO_WINDOW

    match check.output() {
        Ok(output) if output.status.success() => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let mut lines = stdout.lines().map(str::trim);
            let version = lines.next().unwrap_or("").to_string();
            let cuda = lines.next().unwrap_or("");
            PythonEnvStatus {
                ready: true,
                python_path: python.to_string_lossy().to_string(),
                venv_path: venv_path.clone(),
                error: None,
                stage: format!("ready (torch {})", version),
                torch_version: Some(version),
                torch_cuda: !cuda.is_empty(),
                cuda_upgrade_supported: cuda_upgrade_supported(),
            }
        }
        _ => not_ready(
            python.to_string_lossy().to_string(),
            Some("PyTorch not installed in venv".to_string()),
            "deps_missing",
        ),
    }
}

/// What `setup_env` reports while it works.
#[derive(Serialize, Clone, Debug)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SetupUpdate {
    /// A new step started
    Step { step: u32, total: u32, label: String },
    /// Bytes downloaded so far of the packages this step downloads
    Download { done: u64, total: u64 },
    /// A line of output from uv or Python
    Log { line: String },
}

/// "495.1MiB" → bytes
fn parse_size(s: &str) -> Option<u64> {
    let s = s.trim();
    let split = s.find(|c: char| c.is_ascii_alphabetic())?;
    let (num, unit) = s.split_at(split);
    let n: f64 = num.trim().parse().ok()?;
    let mult = match unit {
        "B" => 1.0,
        "KiB" => 1024.0,
        "MiB" => 1024.0 * 1024.0,
        "GiB" => 1024.0 * 1024.0 * 1024.0,
        _ => return None,
    };
    Some((n * mult) as u64)
}

/// Adds up uv's download lines: "Downloading torch (514.2MiB)" when a big download starts
/// and " Downloaded torch" when it finishes. uv only names downloads over about a megabyte,
/// so this tracks the big ones, which are nearly all of the bytes.
#[derive(Default)]
struct DownloadTally {
    sizes: HashMap<String, u64>,
    total: u64,
    done: u64,
}

impl DownloadTally {
    /// Returns (done, total) when the line changed them.
    fn line(&mut self, line: &str) -> Option<(u64, u64)> {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("Downloading ") {
            let (name, size) = rest.split_once(" (")?;
            let size = parse_size(size.strip_suffix(')')?)?;
            if self.sizes.insert(name.to_string(), size).is_none() {
                self.total += size;
            }
            return Some((self.done, self.total));
        }
        if let Some(name) = line.strip_prefix("Downloaded ") {
            let size = self.sizes.remove(name.trim())?;
            self.done += size;
            return Some((self.done, self.total));
        }
        None
    }
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
        // uv writes its progress to stderr
        let err_reader = scope.spawn(|| {
            let mut tally = DownloadTally::default();
            let mut tail: VecDeque<String> = VecDeque::new();
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                if let Some((done, total)) = tally.line(&line) {
                    report(SetupUpdate::Download { done, total });
                }
                report(SetupUpdate::Log { line: line.clone() });
                tail.push_back(line);
                if tail.len() > 20 {
                    tail.pop_front();
                }
            }
            Vec::from(tail).join("\n")
        });
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            report(SetupUpdate::Log { line });
        }
        err_reader.join().unwrap_or_default()
    });

    let status = child.wait().map_err(|e| e.to_string())?;
    Ok((status.success(), stderr_tail))
}

/// `uv pip install` into the venv.
fn uv_pip_install(
    uv: &Path,
    app_data: &Path,
    args: &[&str],
    report: &(dyn Fn(SetupUpdate) + Sync),
) -> Result<(bool, String), String> {
    let mut cmd = uv_command(uv, app_data);
    cmd.args(["pip", "install", "--python"]).arg(venv_python(app_data)).args(args);
    run_streamed(&mut cmd, report)
}

fn require_uv() -> Result<PathBuf, String> {
    find_uv().ok_or_else(|| {
        "TensorBloom's installer tool (uv) is missing from this copy of the app. Reinstall TensorBloom, or if you built it from source, run `node scripts/fetch-uv.mjs` in the app folder.".to_string()
    })
}

const SETUP_STEPS: u32 = 5;

/// Set up Python and PyTorch, reporting each step and uv's output through `report`.
pub fn setup_env(app_data: &Path, report: &(dyn Fn(SetupUpdate) + Sync)) -> Result<PythonEnvStatus, String> {
    let step = |n: u32, label: &str| report(SetupUpdate::Step { step: n, total: SETUP_STEPS, label: label.to_string() });
    let uv = require_uv()?;
    let venv = venv_dir(app_data);
    let python = venv_python(app_data);

    // 1–2. A Python of TensorBloom's own, and a venv made with it. A venv left by an
    // earlier version (made with the system Python) is kept as it is.
    if !python.exists() {
        step(1, "Downloading Python");
        let (ok, stderr) = run_streamed(uv_command(&uv, app_data).args(["python", "install", PYTHON_VERSION]), report)
            .map_err(|e| format!("Couldn't run uv: {}", e))?;
        if !ok {
            return Err(format!("Couldn't download Python: {}", stderr));
        }

        step(2, "Creating the Python environment");
        let (ok, stderr) = run_streamed(
            uv_command(&uv, app_data).args(["venv", "--python", PYTHON_VERSION]).arg(&venv),
            report,
        )
        .map_err(|e| format!("Couldn't run uv: {}", e))?;
        if !ok {
            return Err(format!("Couldn't create the Python environment: {}", stderr));
        }
    }

    // 3. PyTorch, CPU build. The PyTorch index serves torch and its dependencies,
    // so it replaces PyPI for this step rather than being added to it.
    step(3, "Downloading PyTorch");
    let mut torch_args: Vec<&str> = TORCH_PACKAGES.to_vec();
    if cfg!(not(target_os = "macos")) {
        torch_args.extend(["--index-url", TORCH_CPU_INDEX]);
    }
    let (ok, stderr) = uv_pip_install(&uv, app_data, &torch_args, report)
        .map_err(|e| format!("Couldn't run uv: {}", e))?;
    if !ok {
        return Err(format!("Failed to install PyTorch: {}", stderr));
    }

    // 4. The remaining dependencies, from PyPI
    step(4, "Installing the other packages");
    let (ok, stderr) = uv_pip_install(&uv, app_data, &OTHER_PACKAGES, report)
        .map_err(|e| format!("Couldn't run uv: {}", e))?;
    if !ok {
        return Err(format!("Failed to install dependencies: {}", stderr));
    }

    // 5. Make sure torch imports
    step(5, "Checking the install");
    Ok(check_env(app_data))
}

const GPU_STEPS: u32 = 2;

/// Swap the CPU build of PyTorch for the CUDA build, so training can use an NVIDIA GPU.
pub fn install_cuda_torch(app_data: &Path, report: &(dyn Fn(SetupUpdate) + Sync)) -> Result<PythonEnvStatus, String> {
    if !cuda_upgrade_supported() {
        return Err("GPU support with CUDA is only available on Windows and Linux.".to_string());
    }
    let uv = require_uv()?;
    if !venv_python(app_data).exists() {
        return Err("Set up Python first.".to_string());
    }
    let step = |n: u32, label: &str| report(SetupUpdate::Step { step: n, total: GPU_STEPS, label: label.to_string() });

    step(1, "Downloading PyTorch with GPU support");
    // The installed CPU build already satisfies "torch", so ask for these to be replaced
    let mut args: Vec<&str> = TORCH_PACKAGES.to_vec();
    for p in TORCH_PACKAGES {
        args.extend(["--reinstall-package", p]);
    }
    args.extend(["--index-url", TORCH_CUDA_INDEX]);
    let (ok, stderr) = uv_pip_install(&uv, app_data, &args, report)
        .map_err(|e| format!("Couldn't run uv: {}", e))?;
    if !ok {
        return Err(format!("Failed to install PyTorch with GPU support: {}", stderr));
    }

    step(2, "Checking the install");
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
    use super::{parse_size, DownloadTally};

    #[test]
    fn reads_uv_sizes() {
        assert_eq!(parse_size("2.0MiB"), Some(2 * 1024 * 1024));
        assert_eq!(parse_size("1.5GiB"), Some(1536 * 1024 * 1024));
        assert_eq!(parse_size("512B"), Some(512));
        assert_eq!(parse_size("12 parsecs"), None);
    }

    #[test]
    fn tallies_uv_downloads() {
        let mut t = DownloadTally::default();
        assert_eq!(t.line("Resolved 38 packages in 3.40s"), None);
        assert_eq!(t.line("Downloading torch (512.0MiB)"), Some((0, 512 << 20)));
        assert_eq!(t.line("Downloading numpy (16.0MiB)"), Some((0, 528 << 20)));
        assert_eq!(t.line(" Downloaded numpy"), Some((16 << 20, 528 << 20)));
        // A finish for something never announced changes nothing
        assert_eq!(t.line(" Downloaded pillow"), None);
        assert_eq!(t.line(" Downloaded torch"), Some((528 << 20, 528 << 20)));
    }
}
