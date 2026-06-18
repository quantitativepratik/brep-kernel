# Security Policy

This project is a research/prototype geometry kernel, not production CAD infrastructure.

## Supported Versions

Only the current `main` branch is supported for security fixes.

## Reporting A Vulnerability

Please open a private security advisory on GitHub if available. If private advisories are not enabled for the repository, open an issue with a minimal description and avoid posting exploit details publicly.

Useful reports include:

- malformed input that causes a panic in public APIs
- denial-of-service cases from unexpectedly expensive geometry inputs
- memory-safety issues in dependencies or unsafe code paths

The crate currently forbids unsafe operations in unsafe functions and does not contain project-authored unsafe blocks.
