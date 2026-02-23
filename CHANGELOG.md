# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- add CIFAR-10/100 support at 32×32 native resolution
- add timm weight sources, pretrained trial logging, and update docs
- add explicit warnings for GPU-to-CPU device fallback
- add dataset size validation and robust device comparison
- add optimizer_type config field and Bandit quality badge

### Changed

- remove domain-specific naming from framework code
- modernize type hints with PEP 585 and __future__ annotations
- rename models/ to architectures/ and RunPaths models to checkpoints

### Documentation

- update changelog for v0.1.5
- fix Dockerfile entry point and update project documentation
- reorganize README badges and move SonarCloud metrics to TESTING.md

### Fixed

- strict config validation and cleanup
- audit hardening — config wiring, validation, and constant extraction
- complete return type annotations and harden Bandit in CI
- audit hardening — scheduler bug, config wiring, and dead code removal
- audit hardening — trainer monitor_metric, dead code removal, and export wiring
- audit hardening — dead code removal, deduplication, and metric constants

### Miscellaneous

- skip release commits from changelog

## [0.1.5] - 2026-02-20

### Added

- add `orchard init` command for recipe generation

### Changed

- harden type safety, domain-aware transforms, and log coherence

## [0.1.4] - 2026-02-19

### Added

- extend ColorFormatter with header and early-stopping coloring
- add 64x64 resolution support and fix Optuna metric leak

### Changed

- clean up reproducibility module and standardize recipes
- extract shared class-label constants in medical.py

### Fixed

- sanitize timm/ slash in artifact paths and polish v0.1.3
- remove last from_args() references and clean stale artifacts

## [0.1.3] - 2026-02-19

### Build

- bump version to 0.1.3 (0.1.2 burned on PyPI)

## [0.1.2] - 2026-02-19

### Added

- add rank-aware orchestration and normalize relative imports
- add Typer CLI entry point and modernize project tooling
- add model_pool to constrain Optuna architecture search
- consolidate v0.1.2 — remove forge.py, slim config layer, harden CI

### Documentation

- improve readability and fix outdated references
- clarify install order, add NPZ data format note
- fix MiniCNN CPU timing, update metadata docstrings, remove stale roadmap entry

### Fixed

- strip ANSI codes in CLI help test to fix Rich markup assertion

## [0.1.1] - 2026-02-18

### Build

- bump minimum Pillow and psutil versions
- bump version to 0.1.1

### Documentation

- comprehensive documentation overhaul and codebase polish
- convert markdown headings to HTML and update test counts

### Fixed

- harden reproducibility, training safety, and type correctness
- normalize docstring style and strengthen pre-commit hooks
- add build and ci groups to git-cliff commit parsers
- improve logging correctness, public API, and IO configurability


## [0.1.0] - 2026-02-15

First public release of Orchard ML — a type-safe, reproducible deep learning
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
- Modularized monolithic training script into 7-phase RootOrchestrator lifecycle
- Replaced Excel report generation with openpyxl for correct integer formatting

### Fixed

- ONNX export using effective input channels to match model architecture
- Docker shared memory allocation for DataLoader workers
- CUDA determinism with proper seed propagation
- AMP graceful fallback on CPU-only environments
- Channel mismatch and AUC calculation for binary classification
