mod icon_data;
mod python_env;
mod sidecar_manager;

use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use serde::Serialize;
use tauri::{Emitter, Manager};

/// Check if the Python environment is ready.
#[tauri::command]
async fn check_python_env(app: tauri::AppHandle) -> Result<python_env::PythonEnvStatus, String> {
    let app_data = app.path().app_data_dir().map_err(|e| e.to_string())?;
    std::fs::create_dir_all(&app_data).map_err(|e| e.to_string())?;
    Ok(python_env::check_env(&app_data))
}

static SETUP_RUNNING: AtomicBool = AtomicBool::new(false);

/// Set up the Python venv and install PyTorch. Progress arrives as
/// "python-setup" events while it runs.
#[tauri::command]
async fn setup_python_env(app: tauri::AppHandle) -> Result<python_env::PythonEnvStatus, String> {
    let app_data = app.path().app_data_dir().map_err(|e| e.to_string())?;
    std::fs::create_dir_all(&app_data).map_err(|e| e.to_string())?;
    if SETUP_RUNNING.swap(true, Ordering::SeqCst) {
        return Err("Python setup is already running".to_string());
    }
    // Installing takes minutes; keep it off the async runtime
    let result = tauri::async_runtime::spawn_blocking(move || {
        let report = |update: python_env::SetupUpdate| {
            let _ = app.emit("python-setup", update);
        };
        python_env::setup_env(&app_data, &report)
    })
    .await
    .map_err(|e| e.to_string())
    .and_then(|r| r);
    SETUP_RUNNING.store(false, Ordering::SeqCst);
    result
}

/// Get the venv Python path.
#[tauri::command]
async fn get_python_path(app: tauri::AppHandle) -> Result<String, String> {
    let app_data = app.path().app_data_dir().map_err(|e| e.to_string())?;
    python_env::get_python_path(&app_data)
        .ok_or_else(|| "Python environment not set up".to_string())
}

#[derive(Serialize)]
struct GpuDetectResult {
    name: String,
    memory_total_mb: u64,
}

/// Detect GPU via nvidia-smi (independent of Python/sidecar).
#[tauri::command]
async fn detect_gpu() -> Result<Option<GpuDetectResult>, String> {
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
        .output();

    match output {
        Ok(out) if out.status.success() => {
            let stdout = String::from_utf8_lossy(&out.stdout);
            let line = stdout.trim();
            if line.is_empty() {
                return Ok(None);
            }
            // Format: "NVIDIA GeForce RTX 4090, 23028"
            let parts: Vec<&str> = line.splitn(2, ',').collect();
            let name = parts.first().map(|s| s.trim().to_string()).unwrap_or_default();
            let mem: u64 = parts.get(1)
                .and_then(|s| s.trim().parse().ok())
                .unwrap_or(0);
            Ok(Some(GpuDetectResult { name, memory_total_mb: mem }))
        }
        _ => Ok(None),
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .invoke_handler(tauri::generate_handler![
            check_python_env,
            setup_python_env,
            get_python_path,
            detect_gpu,
            sidecar_manager::start_sidecar,
            sidecar_manager::sidecar_write,
            sidecar_manager::stop_sidecar,
        ])
        .setup(|app| {
            // Set the window icon from pre-decoded RGBA data
            if let Some(window) = app.get_webview_window("main") {
                let icon = tauri::image::Image::new(
                    icon_data::ICON_RGBA,
                    icon_data::ICON_WIDTH,
                    icon_data::ICON_HEIGHT,
                );
                let _ = window.set_icon(icon);
            }
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
