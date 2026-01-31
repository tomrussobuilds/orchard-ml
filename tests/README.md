← [Back to Main README](../README.md)

# Test Suite

VisionForge's comprehensive testing infrastructure ensures reliability and maintainability across all components.

## Test Organization

```
tests/                          # Test suite (800+ tests, →100% coverage)
├── smoke_test.py               # 1-epoch E2E verification (~30s)
├── health_check.py             # Dataset integrity validation
├── test_config/                # Config engine tests (210 tests)
├── test_core/                  # Core utilities tests (95 tests)
├── test_data_handler/          # Data loading tests (87 tests)
├── test_evaluation/            # Metrics & viz tests (63 tests)
├── test_models/                # Architecture tests (44 tests)
├── test_optimization/          # Optuna integration tests (78 tests)
└── test_trainer/               # Training loop tests (142 tests)
```

## 🧪 Testing & Quality Assurance

### Test Suite

VisionForge includes a comprehensive test suite with **800+ tests** targeting **→100% code coverage**:

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

- **Unit Tests** (650+ tests): Config validation, metadata injection, type safety
- **Integration Tests** (150+ tests): End-to-end pipeline validation, YAML hydration
- **Smoke Tests**: 1-epoch sanity checks (~30 seconds)
- **Health Checks**: Dataset integrity

### Continuous Integration

GitHub Actions automatically run on every push:

- ✅ **Code Quality**: Black, isort, Flake8 formatting and linting checks
- ✅ **Multi-Python Testing**: Unit tests across Python 3.10, 3.11, 3.12 (800+ tests)
- ✅ **Smoke Test**: 1-epoch end-to-end validation (~30s, CPU-only)
- ✅ **Documentation**: README.md presence verification
- ✅ **Security Scanning**: Bandit (code analysis) and Safety (dependency vulnerabilities)
- ✅ **Code Coverage**: Automated reporting to Codecov (99%+ coverage)

**Pipeline Status:**

| Job | Description | Status |
|-----|-------------|--------|
| **Code Quality** | Black, isort, Flake8 | Continue-on-error (advisory) |
| **Pytest Suite** | 800+ tests, 3 Python versions | ✅ Required to pass |
| **Smoke Test** | 1-epoch E2E validation | ✅ Required to pass |
| **Documentation** | README verification | ✅ Required to pass |
| **Security Scan** | Bandit + Safety | Continue-on-error (advisory) |
| **Build Status** | Aggregate summary | ✅ Fails if pytest or smoke test fails |

View the latest build: [![CI/CD](https://github.com/tomrussobuilds/visionforge/actions/workflows/ci.yml/badge.svg)](https://github.com/tomrussobuilds/visionforge/actions/workflows/ci.yml)

> **Note**: Health checks are not run in CI to avoid excessive dataset downloads. Run locally with `python -m tests.health_check` for dataset integrity validation.
