//! The in-app assistant's connection to the Anthropic API.
//!
//! The user's API key lives in the OS keychain and never reaches the webview: the frontend sends a
//! Messages API request body here, and this module adds the key and makes the call.

use serde_json::Value;
use std::time::Duration;

const KEYCHAIN_SERVICE: &str = "TensorBloom";
const KEYCHAIN_USER: &str = "anthropic-api-key";
const MESSAGES_URL: &str = "https://api.anthropic.com/v1/messages";
const API_VERSION: &str = "2023-06-01";

fn entry() -> Result<keyring::Entry, String> {
    keyring::Entry::new(KEYCHAIN_SERVICE, KEYCHAIN_USER).map_err(|e| format!("Couldn't open the system keychain: {e}"))
}

/// The saved key, else ANTHROPIC_API_KEY from the environment (handy when developing)
fn api_key() -> Option<String> {
    if let Ok(key) = entry().and_then(|e| e.get_password().map_err(|e| e.to_string())) {
        if !key.trim().is_empty() {
            return Some(key);
        }
    }
    std::env::var("ANTHROPIC_API_KEY").ok().filter(|k| !k.trim().is_empty())
}

#[derive(serde::Serialize)]
pub struct KeyStatus {
    configured: bool,
    /// "keychain" or "environment"
    source: Option<&'static str>,
}

/// Whether an API key is available, and where it comes from. Never returns the key itself.
#[tauri::command]
pub async fn assistant_key_status() -> KeyStatus {
    let saved = entry()
        .and_then(|e| e.get_password().map_err(|e| e.to_string()))
        .map(|k| !k.trim().is_empty())
        .unwrap_or(false);
    if saved {
        return KeyStatus { configured: true, source: Some("keychain") };
    }
    let env = std::env::var("ANTHROPIC_API_KEY").map(|k| !k.trim().is_empty()).unwrap_or(false);
    KeyStatus { configured: env, source: env.then_some("environment") }
}

/// Saves the key in the OS keychain
#[tauri::command]
pub async fn assistant_save_key(key: String) -> Result<(), String> {
    let key = key.trim();
    if key.is_empty() {
        return Err("The key is empty.".into());
    }
    if !key.starts_with("sk-ant-") {
        return Err("That doesn't look like an Anthropic API key. Keys start with \"sk-ant-\".".into());
    }
    entry()?.set_password(key).map_err(|e| format!("Couldn't save the key in the system keychain: {e}"))
}

/// Removes the key from the OS keychain
#[tauri::command]
pub async fn assistant_delete_key() -> Result<(), String> {
    match entry()?.delete_credential() {
        Ok(()) | Err(keyring::Error::NoEntry) => Ok(()),
        Err(e) => Err(format!("Couldn't remove the key from the system keychain: {e}")),
    }
}

/// Sends one Messages API request and returns the response body.
///
/// `body` is the request as the API takes it, plus an optional `betas` array that is sent as the
/// anthropic-beta header. Errors come back as a short message, with the API's own message when it gave one.
#[tauri::command]
pub async fn assistant_request(mut body: Value) -> Result<Value, String> {
    let key = api_key().ok_or_else(|| "No API key is set. Add one in the assistant's settings.".to_string())?;
    let model_ok = body.get("model").and_then(Value::as_str).is_some_and(|m| m.starts_with("claude-"));
    if !model_ok {
        return Err("The request has no Claude model.".into());
    }
    let betas = body
        .as_object_mut()
        .and_then(|o| o.remove("betas"))
        .and_then(|b| b.as_array().cloned())
        .unwrap_or_default()
        .iter()
        .filter_map(|b| b.as_str().map(str::to_string))
        .collect::<Vec<_>>();

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(600))
        .build()
        .map_err(|e| e.to_string())?;
    let mut req = client
        .post(MESSAGES_URL)
        .header("x-api-key", key)
        .header("anthropic-version", API_VERSION)
        .header("content-type", "application/json")
        .json(&body);
    if !betas.is_empty() {
        req = req.header("anthropic-beta", betas.join(","));
    }
    let res = req.send().await.map_err(|e| {
        if e.is_timeout() {
            "The request to Claude timed out.".to_string()
        } else if e.is_connect() {
            "Couldn't reach the Anthropic API. Check your internet connection.".to_string()
        } else {
            format!("The request to Claude failed: {e}")
        }
    })?;
    let status = res.status();
    let text = res.text().await.map_err(|e| e.to_string())?;
    let json: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
    if status.is_success() {
        return Ok(json);
    }
    let api_message = json.pointer("/error/message").and_then(Value::as_str).unwrap_or("").to_string();
    Err(match status.as_u16() {
        401 => "The API key was rejected. Check it in the assistant's settings.".to_string(),
        403 => format!("This API key isn't allowed to make that request. {api_message}"),
        429 => "Rate limited by the Anthropic API. Wait a moment and try again.".to_string(),
        529 => "The Anthropic API is overloaded right now. Try again shortly.".to_string(),
        code if api_message.is_empty() => format!("The Anthropic API returned an error ({code})."),
        code => format!("The Anthropic API returned an error ({code}): {api_message}"),
    })
}
