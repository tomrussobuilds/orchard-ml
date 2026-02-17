# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- add ANSI color formatting, fix early-stop trials-saved count
- add dynamic code quality badges via GitHub Gist
- feat(tests): use tmp_path and standardize test paths
- add YAML-configurable search space overrides
- add generic timm backbone integration

### Changed

- extract domain fetchers into fetchers/ sub-package
- add TrackerProtocol, unify environment log sections

### Documentation

- update ConvNeXt-Tiny params to ~27.8M, clarify per-trial timings, add tracking module to package docs and architecture diagram
- streamline OPTIMIZATION.md, remove duplicate sections

### Fixed

- replace --prepend with custom script to prevent duplicate Unreleased sections
- resolve mypy type errors and pin version to 1.19.1

### Miscellaneous

- add automated changelog with git-cliff and pre-commit hook
- add pre-commit linting hooks, centralize flake8 config

## [0.1.0] - 2026-02-15

First public release of VisionForge — a type-safe, reproducible deep learning
framework for computer vision research.

### Added

- **Training framework** with single unified entry point (`forge.py`) controlled by YAML recipes
- **Model architectures**: ResNet-18 (multi-resolution), MiniCNN, EfficientNet-B0, ConvNeXt-Tiny, ViT-Tiny
- **Dataset support**: MedMNIST v2 (medical imaging) and Galaxy10 DECals (astronomical imaging)
- **Type-safe configuration engine** built on Pydantic V2 with hierarchical YAML schemas
- **Hyperparameter optimization** via Optuna with TPE sampling, Median Pruning, and model search
- **ONNX export pipeline** for production deployment across all training recipes
- **MLflow experiment tracking** (opt-in) with local SQLite backend
- **Test-Time Augmentation (TTA)** with adaptive ensemble predictions
- **Reproducibility guarantees**: BLAKE2b-hashed run directories, full YAML config snapshots, seed control
- **Hardware auto-detection** and optimization for CPU, CUDA, and MPS backends
- **Cluster-safe execution** with kernel-level `fcntl` file locking
- **Docker support** with optimized multi-stage builds
- **CI/CD pipeline** with GitHub Actions (pytest, mypy, coverage, pip-audit, SonarQube)
- **Comprehensive test suite** with >90% code coverage

### Changed

- Migrated configuration from dataclasses to Pydantic V2 for runtime validation
- Migrated to Torchvision V2 transforms pipeline
- Modularized monolithic training script into 7-phase atomic initialization lifecycle
- Replaced Excel report generation with openpyxl for correct integer formatting

### Fixed

- ONNX export using effective input channels to match model architecture
- Docker shared memory allocation for DataLoader workers
- CUDA determinism with proper seed propagation
- AMP graceful fallback on CPU-only environments
- Channel mismatch and AUC calculation for binary classification
