# Releasing TensorBloom

Pushing a tag that starts with `v` (for example `v0.2.0`) runs
`.github/workflows/release.yml`. It builds Windows, Linux, and Apple Silicon
macOS installers, signs the update bundles, and publishes them to a GitHub
release together with `latest.json`, which installed apps read to find updates.

## Before the first release

### Updater signing key (required)

Installed apps only accept updates signed with the project's updater key.
Generate it once and keep the private key somewhere safe, such as a password
manager. **If it is lost, installed apps can no longer be updated.**

```sh
cd app
npx tauri signer generate -w ~/.tauri/tensorbloom.key
```

1. Put the contents of `~/.tauri/tensorbloom.key.pub` in
   `plugins.updater.pubkey` in `app/src-tauri/tauri.conf.json`.
2. In the repository's **Settings → Secrets and variables → Actions**, add:
   - `TAURI_SIGNING_PRIVATE_KEY`: the contents of `~/.tauri/tensorbloom.key`
   - `TAURI_SIGNING_PRIVATE_KEY_PASSWORD`: the password you chose (leave it
     out if you chose none)

The release workflow stops with an error if the key is missing.

### Apple signing and notarization (optional)

Without these secrets the Mac app is ad-hoc signed. It runs, but the first
time it opens macOS blocks it, and the user has to allow it under
**System Settings → Privacy & Security → Open Anyway**. With them, the app is
signed with a Developer ID and notarized, and opens normally.

It takes an [Apple Developer Program](https://developer.apple.com/programs/)
membership. Then add these Actions secrets:

| Secret | What it is |
| --- | --- |
| `APPLE_CERTIFICATE` | The "Developer ID Application" certificate exported as `.p12`, base64-encoded (`base64 -i cert.p12 \| pbcopy`) |
| `APPLE_CERTIFICATE_PASSWORD` | The password set when exporting the `.p12` |
| `APPLE_SIGNING_IDENTITY` | The certificate name, such as `Developer ID Application: Your Name (TEAMID)` |
| `APPLE_ID` | The Apple ID email of the developer account |
| `APPLE_PASSWORD` | An [app-specific password](https://support.apple.com/en-us/102654) for that Apple ID |
| `APPLE_TEAM_ID` | The 10-character team ID from the developer account |

## Cutting a release

1. Bump the version in `app/package.json`, `app/src-tauri/Cargo.toml`, and
   `app/src-tauri/tauri.conf.json`. The updater compares this version with
   `latest.json`, so it must go up with every release.
2. Merge to `main`, then tag and push: `git tag v0.2.0 && git push origin v0.2.0`.
3. Check the release page once the workflow finishes, and edit the notes.
   Installed apps show the "TensorBloom 0.2.0 is available" banner on their
   next launch.
