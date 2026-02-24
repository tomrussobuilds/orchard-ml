← [Back to Home](../index.md)

<h1 align="center">Testing & Quality Assurance</h1>

<h2>Environment Verification</h2>

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

<h2>Code Quality Checks</h2>

Orchard ML includes automated quality check scripts that run all code quality tools in sequence.

<h3>Quick Check (Recommended)</h3>

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

<h3>Extended Check (Thorough)</h3>

Comprehensive checks with type checking (~60-120 seconds):

```bash
# Run extended quality checks with MyPy
bash scripts/check_quality_full.sh
```

**Additional checks:**
- **MyPy**: Static type checking
- **Radon**: Extended metrics (raw metrics, detailed analysis)
- **Pytest**: HTML coverage report

<h3>Tool Descriptions</h3>

<h4>Formatting Tools</h4>

- **Black**: Opinionated code formatter (line length: 100)
  ```bash
  black orchard/ tests/  # Auto-fix
  ```

- **isort**: Sorts imports alphabetically and by type
  ```bash
  isort orchard/ tests/  # Auto-fix
  ```

<h4>Linting Tools</h4>

- **Flake8**: PEP 8 style guide enforcement
  - Checks: unused variables, imports, style violations
  - Max line length: 100
  - Ignored: E203, W503
  ```bash
  flake8 orchard/ tests/ --max-line-length=100 --extend-ignore=E203,W503
  ```

<h4>Security Tools</h4>

- **Bandit**: Detects common security issues
  - Checks: hardcoded passwords, SQL injection, insecure temp files
  - Severity: Medium and High only (`-ll`)
  ```bash
  bandit -r orchard/ -ll -q
  ```

<h4>Complexity Analysis</h4>

- **Radon**: Code metrics analyzer
  - **Cyclomatic Complexity (CC)**: Measures code complexity (max: B = 6-10)
  - **Maintainability Index (MI)**: Measures maintainability (min: B = 20-100)
  - Grades: A (best), B, C, D, E, F (worst)
  ```bash
  radon cc orchard/ -n B --total-average  # Complexity
  radon mi orchard/ -n B                   # Maintainability
  ```

<h4>Type Checking</h4>

- **MyPy**: Static type checker for Python
  - Verifies type hints and catches type errors at compile time
  ```bash
  mypy orchard/ --ignore-missing-imports --no-strict-optional
  ```

<h3>Individual Tool Usage</h3>

```bash
# Code formatting check
black --check --diff orchard/ tests/

# Import sorting check
isort --check-only --diff orchard/ tests/

# Linting
flake8 orchard/ tests/ --max-line-length=100 --extend-ignore=E203,W503

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

<h3>Installation</h3>

Install dev dependencies (includes all quality tools):

```bash
pip install -e ".[dev]"
```

---

<h2>Testing & Quality Assurance</h2>

<h3>Test Suite</h3>

Orchard ML includes a comprehensive test suite targeting **→100% code coverage**:

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
- **Health Checks**: Dataset integrity

<h3>Continuous Integration</h3>

GitHub Actions automatically run on every push:

- ✅ **Code Quality**: Black, isort, Flake8, mypy formatting, linting, and type checks
- ✅ **Multi-Python Testing**: Unit tests across Python 3.10–3.14
- ✅ **Smoke Test**: 1-epoch end-to-end validation (~30s, CPU-only)
- ✅ **Documentation**: README.md presence verification
- ✅ **Security Scanning**: Bandit (code analysis) and pip-audit (dependency vulnerabilities)
- ✅ **Code Coverage**: Automated reporting to Codecov (99%+ coverage)
- ✅ **SonarCloud**: Continuous code quality inspection (reliability, security, maintainability)

**SonarCloud Metrics:**

<p>
  <a href="https://sonarcloud.io/summary/new_code?id=tomrussobuilds_orchard-ml"><img src="https://sonarcloud.io/api/project_badges/measure?project=tomrussobuilds_orchard-ml&metric=reliability_rating" alt="Reliability"></a>
  <a href="https://sonarcloud.io/summary/new_code?id=tomrussobuilds_orchard-ml"><img src="https://sonarcloud.io/api/project_badges/measure?project=tomrussobuilds_orchard-ml&metric=security_rating" alt="Security"></a>
  <a href="https://sonarcloud.io/summary/new_code?id=tomrussobuilds_orchard-ml"><img src="https://sonarcloud.io/api/project_badges/measure?project=tomrussobuilds_orchard-ml&metric=sqale_rating" alt="Maintainability"></a>
  <a href="https://sonarcloud.io/summary/new_code?id=tomrussobuilds_orchard-ml"><img src="https://sonarcloud.io/api/project_badges/measure?project=tomrussobuilds_orchard-ml&metric=coverage" alt="Coverage"></a>
  <a href="https://sonarcloud.io/summary/new_code?id=tomrussobuilds_orchard-ml"><img src="https://sonarcloud.io/api/project_badges/measure?project=tomrussobuilds_orchard-ml&metric=bugs" alt="Bugs"></a>
  <a href="https://sonarcloud.io/summary/new_code?id=tomrussobuilds_orchard-ml"><img src="https://sonarcloud.io/api/project_badges/measure?project=tomrussobuilds_orchard-ml&metric=code_smells" alt="Code Smells"></a>
</p>

> All badges above are dynamic and updated automatically by SonarCloud on every push to `main`.

**Pipeline Status:**

| Job | Description | Status |
|-----|-------------|--------|
| **Code Quality** | Black, isort, Flake8, mypy | ✅ Required to pass |
| **Pytest Suite** | 5 Python versions | ✅ Required to pass |
| **Smoke Test** | 1-epoch E2E validation | ✅ Required to pass |
| **Documentation** | README verification | ✅ Required to pass |
| **Security Scan** | Bandit + pip-audit | ✅ Required to pass (Bandit hard-fail, pip-audit advisory) |
| **Build Status** | Aggregate summary | ✅ Fails if lint, pytest, smoke test, or security fails |

View the latest build: [![CI/CD](https://github.com/tomrussobuilds/orchard-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/tomrussobuilds/orchard-ml/actions/workflows/ci.yml)

> **Note**: Health checks are not run in CI to avoid excessive dataset downloads. Run locally with `python -m tests.health_check` for dataset integrity validation.

> **Note**: Python 3.14 (dev) is tested for core functionality only. ONNX export requires `onnxruntime>=1.24.1` which provides Python 3.14 wheels.

---
