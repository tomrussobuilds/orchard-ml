← [Back to Main README](../README.md)

# Test Suite

Orchard ML's comprehensive testing infrastructure ensures reliability and maintainability across all components.

## Test Organization

```
tests/                          # Test suite (~1,100 tests, 100% coverage)
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
├── test_models/                # Architecture tests (factory, all model builders)
├── test_optimization/          # Optuna integration tests (objective, orchestrator)
├── test_paths/                 # Path management tests (constants, run_paths)
├── test_pipeline/              # Pipeline phase tests
├── test_tracking/              # MLflow tracking tests (tracker, integration)
└── test_trainer/               # Training loop tests (engine, trainer, setup)
```

## Testing & Quality Assurance

### Test Suite

Orchard ML includes a comprehensive test suite with **1,100+ tests** targeting **100% code coverage**:

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

### Test Categories

- **Unit Tests** (~920 tests): Config validation, metadata injection, type safety
- **Integration Tests** (~40 tests): End-to-end pipeline validation, YAML hydration
- **Smoke Tests**: 1-epoch sanity checks (~30 seconds)
- **Health Checks**: Dataset integrity verification

### Continuous Integration

GitHub Actions automatically run on every push:

- ✅ **Code Quality**: Black, isort, Flake8 formatting and linting checks
- ✅ **Multi-Python Testing**: Unit tests across Python 3.10, 3.11, 3.12 (~1,000 tests)
- ✅ **Smoke Test**: 1-epoch end-to-end validation (~30s, CPU-only)
- ✅ **Documentation**: README.md presence verification
- ✅ **Security Scanning**: Bandit (code analysis) and Safety (dependency vulnerabilities)
- ✅ **Code Coverage**: Automated reporting to Codecov (99%+ coverage)

**Pipeline Status:**

| Job | Description | Status |
|-----|-------------|--------|
| **Code Quality** | Black, isort, Flake8 | Continue-on-error (advisory) |
| **Pytest Suite** | ~1,100 tests, 3 Python versions | ✅ Required to pass |
| **Smoke Test** | 1-epoch E2E validation | ✅ Required to pass |
| **Documentation** | README verification | ✅ Required to pass |
| **Security Scan** | Bandit + Safety | Continue-on-error (advisory) |
| **Build Status** | Aggregate summary | ✅ Fails if pytest or smoke test fails |

View the latest build: [![CI/CD](https://github.com/tomrussobuilds/orchard-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/tomrussobuilds/orchard-ml/actions/workflows/ci.yml)

> **Note**: Health checks are not run in CI to avoid excessive dataset downloads. Run locally with `python -m tests.health_check` for dataset integrity validation.

### Additional CI/CD Workflows

Beyond the main CI pipeline, the project includes automated release and publishing workflows:

| Workflow | Trigger | Description |
|----------|---------|-------------|
| **CI/CD** (`ci.yml`) | Every push/PR | Full test suite, code quality, smoke test, security scan |
| **Badges** (`badges.yml`) | Push to main | Updates dynamic quality badges (Black, isort, Flake8, mypy, Radon) |
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
