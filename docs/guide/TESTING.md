← [Back to Main README](../../README.md)

# Testing & Quality Assurance

## Environment Verification

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

## Code Quality Checks

Orchard ML includes automated quality check scripts that run all code quality tools in sequence.

### Quick Check (Recommended)

Fast quality checks for everyday development (~30-60 seconds):

```bash
# Run all standard quality checks
bash scripts/check_quality.sh
```

**What it checks:**
- **Black**: Code formatting compliance (PEP 8 style, 100 chars)
- **isort**: Import statement ordering
- **Flake8**: PEP 8 linting and code smells
- **Bandit**: Security vulnerability scanning
- **Radon**: Cyclomatic complexity & maintainability index
- **Pytest**: Full test suite with coverage report

### Extended Check (Thorough)

Comprehensive checks with type checking (~60-120 seconds):

```bash
# Run extended quality checks with MyPy
bash scripts/check_quality_full.sh
```

**Additional checks:**
- **MyPy**: Static type checking
- **Radon**: Extended metrics (raw metrics, detailed analysis)
- **Pytest**: HTML coverage report

### Tool Descriptions

#### Formatting Tools

- **Black**: Opinionated code formatter (line length: 100)
  ```bash
  black orchard/ tests/ forge.py  # Auto-fix
  ```

- **isort**: Sorts imports alphabetically and by type
  ```bash
  isort orchard/ tests/ forge.py  # Auto-fix
  ```

#### Linting Tools

- **Flake8**: PEP 8 style guide enforcement
  - Checks: unused variables, imports, style violations
  - Max line length: 100
  - Ignored: E203, W503
  ```bash
  flake8 orchard/ tests/ --max-line-length=100 --extend-ignore=E203,W503
  ```

#### Security Tools

- **Bandit**: Detects common security issues
  - Checks: hardcoded passwords, SQL injection, insecure temp files
  - Severity: Medium and High only (`-ll`)
  ```bash
  bandit -r orchard/ -ll -q
  ```

#### Complexity Analysis

- **Radon**: Code metrics analyzer
  - **Cyclomatic Complexity (CC)**: Measures code complexity (max: B = 6-10)
  - **Maintainability Index (MI)**: Measures maintainability (min: B = 20-100)
  - Grades: A (best), B, C, D, E, F (worst)
  ```bash
  radon cc orchard/ -n B --total-average  # Complexity
  radon mi orchard/ -n B                   # Maintainability
  ```

#### Type Checking

- **MyPy**: Static type checker for Python
  - Verifies type hints and catches type errors at compile time
  ```bash
  mypy orchard/ --ignore-missing-imports --no-strict-optional
  ```

### Individual Tool Usage

```bash
# Code formatting check
black --check --diff orchard/ tests/ forge.py

# Import sorting check
isort --check-only --diff orchard/ tests/ forge.py

# Linting
flake8 orchard/ tests/ forge.py --max-line-length=100 --extend-ignore=E203,W503

# Security scanning
bandit -r orchard/ -ll -q

# Complexity analysis
radon cc orchard/ -n B --total-average
radon mi orchard/ -n B

# Type checking
mypy orchard/ --ignore-missing-imports

# Tests with coverage
pytest --cov=orchard --cov-report=term-missing -v tests/
```

### Installation

Install dev dependencies (includes all quality tools):

```bash
pip install -e ".[dev]"
```

---

## Testing & Quality Assurance

### Test Suite

Orchard ML includes a comprehensive test suite with **1,000+ tests** targeting **→100% code coverage**:

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

- **Unit Tests** (850+ tests): Config validation, metadata injection, type safety
- **Integration Tests** (150+ tests): End-to-end pipeline validation, YAML hydration
- **Smoke Tests**: 1-epoch sanity checks (~30 seconds)
- **Health Checks**: Dataset integrity

### Continuous Integration

GitHub Actions automatically run on every push:

- ✅ **Code Quality**: Black, isort, Flake8, mypy formatting, linting, and type checks
- ✅ **Multi-Python Testing**: Unit tests across Python 3.10–3.14 (1,000+ tests)
- ✅ **Smoke Test**: 1-epoch end-to-end validation (~30s, CPU-only)
- ✅ **Documentation**: README.md presence verification
- ✅ **Security Scanning**: Bandit (code analysis) and pip-audit (dependency vulnerabilities)
- ✅ **Code Coverage**: Automated reporting to Codecov (99%+ coverage)
- ✅ **SonarCloud**: Continuous code quality inspection (reliability, security, maintainability)

**Pipeline Status:**

| Job | Description | Status |
|-----|-------------|--------|
| **Code Quality** | Black, isort, Flake8, mypy | Continue-on-error (advisory) |
| **Pytest Suite** | 1,000+ tests, 5 Python versions | ✅ Required to pass |
| **Smoke Test** | 1-epoch E2E validation | ✅ Required to pass |
| **Documentation** | README verification | ✅ Required to pass |
| **Security Scan** | Bandit + pip-audit | Continue-on-error (advisory) |
| **Build Status** | Aggregate summary | ✅ Fails if pytest or smoke test fails |

View the latest build: [![CI/CD](https://github.com/tomrussobuilds/orchard-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/tomrussobuilds/orchard-ml/actions/workflows/ci.yml)

> **Note**: Health checks are not run in CI to avoid excessive dataset downloads. Run locally with `python -m tests.health_check` for dataset integrity validation.

> **Note**: Python 3.14 (dev) is tested for core functionality only. ONNX export is not supported on 3.14-dev as `onnxruntime` does not yet provide compatible wheels.

---
