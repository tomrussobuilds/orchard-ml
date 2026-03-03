# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 0.1.x   | Yes                |
| < 0.1   | No                 |

We recommend running the latest release or installing from the `main` branch
to ensure you have the most recent security fixes.

## Reporting a Vulnerability

If you discover a security vulnerability in Orchard ML, **please do not open a public issue.**

Instead, report it privately via
[GitHub Security Advisories](https://github.com/tomrussobuilds/orchard-ml/security/advisories/new).

Please include:

- A description of the vulnerability and its potential impact.
- Steps to reproduce (minimal recipe YAML, code snippet, or command).
- The version of Orchard ML and Python you are using.

You can expect an initial response within **7 days**. Once confirmed, a fix
will be released as a patch version and credited in the changelog unless you
prefer to remain anonymous.

## Scope

The following are **in scope**:

- The `orchard` Python package published on PyPI.
- Official Docker images and CI workflows in this repository.
- The documentation site at <https://tomrussobuilds.github.io/orchard-ml/>.

The following are **out of scope**:

- Vulnerabilities in upstream dependencies (PyTorch, NumPy, etc.) — please
  report those to the respective projects.
- Issues that require physical access to the host machine.
- Denial-of-service attacks against the documentation site.

## Security Practices

This project enforces the following measures in CI:

- **Static analysis** — Bandit (security linter) and Ruff on every push.
- **SAST** — GitHub CodeQL analysis on push, pull request, and weekly schedule.
- **Dependency auditing** — pip-audit for known vulnerabilities.
- **Supply chain** — PyPI publishing via OIDC trusted publishing (no long-lived secrets).
- **Type safety** — mypy in strict mode across the entire codebase.
- **Docker** — Non-root container user, pinned CUDA base image.
