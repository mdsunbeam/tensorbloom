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
- Does not connect to external servers (except for dataset downloads)

The Python sidecar executes arbitrary PyTorch code by design (model compilation, training). This is expected behavior, not a vulnerability.
