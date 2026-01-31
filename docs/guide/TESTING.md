← [Back to Main README](../../README.md)

# Testing & Quality Assurance

## ✅ Environment Verification

**Smoke Test** (1-epoch sanity check):
```bash
# Default: BloodMNIST 28×28
python -m tests.smoke_test

# Custom dataset
python -m tests.smoke_test --dataset pathmnist
```

**Output:** Validates full pipeline in <30 seconds:
- Dataset loading and preprocessing
- Model instantiation and weight transfer
- Training loop execution
- Evaluation metrics computation
- Excel/PNG artifact generation

**Health Check** (dataset integrity):
```bash
python -m tests.health_check --dataset organcmnist --resolution 224
```

**Output:** Verifies:
- MD5 checksum matching
- NPZ key structure (`train_images`, `train_labels`, `val_images`, etc.)
- Sample count validation

---

## 🔧 Code Quality Checks

VisionForge includes an automated quality check script that runs all code quality tools in sequence:

```bash
# Run all quality checks at once
bash scripts/check_quality.sh
```

**What it checks:**
- **Black**: Code formatting compliance
- **isort**: Import statement ordering
- **Flake8**: PEP 8 linting and code smells
- **Pytest**: Full test suite with coverage report

**Individual checks:**
```bash
# Code formatting
black --check orchard/ tests/ main.py optimize.py

# Import sorting
isort --check-only orchard/ tests/ main.py optimize.py

# Linting
flake8 orchard/ tests/ main.py optimize.py --max-line-length=100 --extend-ignore=E203,W503

# Tests with coverage
pytest --cov=orchard --cov-report=term-missing -v tests/
```

---

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

---
