# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

1. **Do not** open a public issue
2. Email **mdsunbeam3.14@gmail.com** with:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
3. You will receive a response within 48 hours

## Scope

TensorBloom is a desktop application that:
- Executes Python code via a local sidecar process
- Reads/writes files from the local filesystem
- Connects to external servers only for:
  - Setting up Python and PyTorch on first use (a standalone Python build that uv downloads, plus packages from PyPI and the PyTorch package index)
  - Dataset downloads, and Hugging Face when you import, chat with or fine-tune a language model
  - The assistant, which is off until you add your own Anthropic API key. Requests go to the Anthropic API, and **See what's sent** in the panel shows what a message includes. The key is kept in the OS keychain and never reaches the interface or project files.
  - Checking GitHub releases for updates. Update bundles are signed, and the app checks the signature before installing.

The app sends no usage analytics. **Help > Send Feedback** opens a GitHub issue form in your browser and sends nothing on its own.

The Python sidecar executes arbitrary PyTorch code by design (model compilation, training). This is expected behavior, not a vulnerability.
