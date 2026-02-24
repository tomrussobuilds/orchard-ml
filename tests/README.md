← [Back to Main README](../README.md)

<h1 align="center">Test Suite</h1>

Orchard ML's comprehensive testing infrastructure ensures reliability and maintainability across all components.

<h2>Test Organization</h2>

```
tests/                          # Test suite (100% coverage)
├── conftest.py                 # Shared pytest fixtures
├── smoke_test.py               # 1-epoch E2E verification (~30s)
├── health_check.py             # Dataset integrity validation
├── test_config/                # Config engine tests (manifest, sub-configs, types)
├── test_core/                  # Core utilities tests (CLI, orchestrator, metadata)
├── test_data_handler/          # Data loading tests (fetcher, dataset, transforms)
├── test_environment/           # Environment tests (hardware, reproducibility, guards)
├── test_evaluation/            # Metrics & viz tests (evaluator, TTA, reporting)
├── test_export/                # Export tests (ONNX exporter, validation)
├── test_io/                    # I/O tests (serialization, checkpoints)
├── test_logger/                # Logging tests (logger, reporter)
├── test_architectures/         # Architecture tests (factory, all model builders)
├── test_optimization/          # Optuna integration tests (objective, orchestrator)
├── test_paths/                 # Path management tests (constants, run_paths)
├── test_pipeline/              # Pipeline phase tests
├── test_tracking/              # MLflow tracking tests (tracker, integration)
└── test_trainer/               # Training loop tests (engine, trainer, setup)
```

<h2>Testing & Quality Assurance</h2>

<h3>Test Suite</h3>

Orchard ML includes a comprehensive test suite targeting **100% code coverage**:

<p>
  <img src="https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/tomrussobuilds/7835190af6011e9051b673c8be974f8a/raw/tests_unit.json" alt="Unit Tests">
  <img src="https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/tomrussobuilds/7835190af6011e9051b673c8be974f8a/raw/tests_integration.json" alt="Integration Tests">
</p>

```bash
# Run full test suite
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=orchard --cov-report=html

# Run specific test categories
pytest tests/ -m unit          # Unit tests only
pytest tests/ -m integration   # Integration tests only

# Run parallel tests (faster)
pytest tests/ -n auto
```

<h3>Test Categories</h3>

- **Unit Tests**: Config validation, metadata injection, type safety
- **Integration Tests**: End-to-end pipeline validation, YAML hydration
- **Smoke Tests**: 1-epoch sanity checks (~30 seconds)
- **Health Checks**: Dataset integrity verification

<h3>Continuous Integration</h3>

GitHub Actions automatically run on every push:

- ✅ **Code Quality**: Black, isort, Flake8 formatting and linting checks
- ✅ **Multi-Python Testing**: Unit tests across Python 3.10–3.14
- ✅ **Smoke Test**: 1-epoch end-to-end validation (~30s, CPU-only)
- ✅ **Documentation**: README.md presence verification
- ✅ **Security Scanning**: Bandit (code analysis) and pip-audit (dependency vulnerabilities)
- ✅ **Code Coverage**: Automated reporting to Codecov (99%+ coverage)

**Pipeline Status:**

| Job | Description | Status |
|-----|-------------|--------|
| **Code Quality** | Black, isort, Flake8, mypy | ✅ Required to pass |
| **Pytest Suite** | 5 Python versions + SonarCloud | ✅ Required to pass |
| **Smoke Test** | 1-epoch E2E validation | ✅ Required to pass |
| **Documentation** | MkDocs build + GitHub Pages deploy | ✅ Main branch only |
| **Security Scan** | Bandit (hard-fail) + pip-audit (advisory) | ✅ Required to pass |
| **Build Status** | Aggregate summary | ✅ Fails if lint, pytest, smoke, or security fails |

View the latest build: [![CI/CD](https://github.com/tomrussobuilds/orchard-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/tomrussobuilds/orchard-ml/actions/workflows/ci.yml)

> **Note**: Health checks are not run in CI to avoid excessive dataset downloads. Run locally with `python -m tests.health_check` for dataset integrity validation.
>
> For detailed guides on smoke tests and health checks, see [docs/guide/TESTING.md](../docs/guide/TESTING.md).

<h3>Additional CI/CD Workflows</h3>

Beyond the main CI pipeline, the project includes automated release and publishing workflows:

| Workflow | Trigger | Description |
|----------|---------|-------------|
| **CI/CD** (`ci.yml`) | Every push/PR | Full test suite, code quality, smoke test, security scan, docs deploy |
| **Badges** (`badges.yml`) | Push to main | Updates dynamic quality badges (Black, isort, Flake8, mypy, Radon) |
| **Documentation** (`docs.yml`) | Manual dispatch | MkDocs build + GitHub Pages deploy (standalone re-deploy) |
| **Release** (`release.yml`) | Tag push (`v*`) | Creates GitHub Release with auto-generated changelog (git-cliff) |
| **Publish** (`publish.yml`) | Tag push (`v*`) | Builds and publishes package to PyPI via Trusted Publisher |

**Release process:**
```bash
# Tag a new version and push to trigger release + publish
git tag v0.2.0
git push origin v0.2.0
# → GitHub Release created automatically with changelog
# → Package built and published to PyPI
```
