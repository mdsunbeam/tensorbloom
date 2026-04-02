//! Manages the Python sidecar process lifecycle.
//! Spawns the sidecar using the venv Python, and provides stdin/stdout piping
//! through Tauri commands.

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
use tauri::{AppHandle, Emitter, Manager};

#[cfg(windows)]
use std::os::windows::process::CommandExt;

static SIDECAR: Mutex<Option<SidecarState>> = Mutex::new(None);

struct SidecarState {
    child: Child,
    stdin_writer: Arc<Mutex<Box<dyn Write + Send>>>,
}

/// Start the sidecar process. Returns once the process is spawned.
#[tauri::command]
pub async fn start_sidecar(app: AppHandle) -> Result<String, String> {
    let mut guard = SIDECAR.lock().map_err(|e| e.to_string())?;

    // Already running?
    if let Some(ref mut state) = *guard {
        if state.child.try_wait().map_err(|e| e.to_string())?.is_none() {
            return Ok("already_running".to_string());
        }
    }

    // Get paths
    let app_data = app.path().app_data_dir().map_err(|e| e.to_string())?;
    let python_path = crate::python_env::get_python_path(&app_data)
        .ok_or("Python environment not set up")?;
    let script_path = get_script_path(&app)?;

    // Spawn the process
    let mut cmd = Command::new(&python_path);
    cmd.arg("-u")  // unbuffered
        .arg(&script_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    #[cfg(windows)]
    cmd.creation_flags(0x08000000); // CREATE_NO_WINDOW

    let mut child = cmd.spawn()
        .map_err(|e| format!("Failed to spawn sidecar: {}", e))?;

    let stdin = child.stdin.take().ok_or("Failed to get stdin")?;
    let stdout = child.stdout.take().ok_or("Failed to get stdout")?;
    let stderr = child.stderr.take().ok_or("Failed to get stderr")?;

    let stdin_writer: Arc<Mutex<Box<dyn Write + Send>>> = Arc::new(Mutex::new(Box::new(stdin)));

    // Read stdout in a thread, emit events to the frontend
    let app_handle = app.clone();
    thread::spawn(move || {
        let reader = BufReader::new(stdout);
        for line_result in reader.lines() {
            let line: String = match line_result {
                Ok(l) => l,
                Err(_) => break,
            };
            if !line.trim().is_empty() {
                let _ = app_handle.emit("sidecar-stdout", &line);
            }
        }
    });

    // Read stderr in a thread
    let app_handle2 = app.clone();
    thread::spawn(move || {
        let reader = BufReader::new(stderr);
        for line_result in reader.lines() {
            let line: String = match line_result {
                Ok(l) => l,
                Err(_) => break,
            };
            if !line.trim().is_empty() {
                let _ = app_handle2.emit("sidecar-stderr", &line);
            }
        }
    });

    *guard = Some(SidecarState {
        child,
        stdin_writer,
    });

    Ok("started".to_string())
}

/// Write a line to the sidecar's stdin.
#[tauri::command]
pub async fn sidecar_write(data: String) -> Result<(), String> {
    let guard = SIDECAR.lock().map_err(|e| e.to_string())?;
    if let Some(ref state) = *guard {
        let mut writer = state.stdin_writer.lock().map_err(|e| e.to_string())?;
        writer
            .write_all(format!("{}\n", data).as_bytes())
            .map_err(|e| format!("Write failed: {}", e))?;
        writer.flush().map_err(|e| format!("Flush failed: {}", e))?;
        Ok(())
    } else {
        Err("Sidecar not running".to_string())
    }
}

/// Stop the sidecar process.
#[tauri::command]
pub async fn stop_sidecar() -> Result<(), String> {
    let mut guard = SIDECAR.lock().map_err(|e| e.to_string())?;
    if let Some(ref mut state) = *guard {
        let _ = state.child.kill();
        let _ = state.child.wait();
    }
    *guard = None;
    Ok(())
}

fn get_script_path(app: &AppHandle) -> Result<String, String> {
    // Try resource dir first (production)
    if let Ok(resource_dir) = app.path().resource_dir() {
        // Mapped resources: python/tensorbloom_sidecar.py
        let script = resource_dir.join("python").join("tensorbloom_sidecar.py");
        if script.exists() {
            return Ok(script.to_string_lossy().to_string());
        }
        // Flat resources: tensorbloom_sidecar.py directly in resource dir
        let script_flat = resource_dir.join("tensorbloom_sidecar.py");
        if script_flat.exists() {
            return Ok(script_flat.to_string_lossy().to_string());
        }
    }

    // Dev mode: try CWD (varies depending on how the app is launched)
    let cwd_candidates = [
        "python/tensorbloom_sidecar.py",
        "../python/tensorbloom_sidecar.py",
    ];
    for candidate in &cwd_candidates {
        let p = std::path::Path::new(candidate);
        if p.exists() {
            return Ok(p.canonicalize().map_err(|e| e.to_string())?.to_string_lossy().to_string());
        }
    }

    // Dev mode: relative to exe location (app/src-tauri/target/debug/)
    if let Ok(exe) = std::env::current_exe() {
        if let Some(exe_dir) = exe.parent() {
            // Walk up from the exe dir trying to find the python/ directory
            let mut dir = exe_dir.to_path_buf();
            for _ in 0..6 {
                let candidate = dir.join("python").join("tensorbloom_sidecar.py");
                if candidate.exists() {
                    return Ok(candidate.canonicalize().map_err(|e| e.to_string())?.to_string_lossy().to_string());
                }
                if !dir.pop() { break; }
            }
        }
    }

    Err(format!(
        "Could not find tensorbloom_sidecar.py (cwd: {:?}, exe: {:?})",
        std::env::current_dir().ok(),
        std::env::current_exe().ok()
    ))
}

