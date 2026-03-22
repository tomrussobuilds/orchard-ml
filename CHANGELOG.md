# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- add detection task adapters and register detection task
- add detection task adapters and register detection task
- add detection data loading pipeline
- add Faster R-CNN builder and register in model factory
- add detection config validation, ONNX guard, recipe, and CLI init
- wire DetectionDataset into DataLoaderFactory
- Optuna detection guard + remove unused in_channels from fasterrcnn
- add --task flag to orchard init for detection recipes
- register PennFudan detection dataset in metadata registry
- add PennFudan detection fetcher and wire end-to-end pipeline
- add structured report to detection pipeline and harden mutation coverage

### Changed

- widen TaskTrainingStep protocol for detection support
- extract shared to_cpu helper, export detection classes, update plan
- extract _forward_and_loss to reduce train_one_epoch complexity
- extract export validation and benchmark helpers from run_export_phase
- reorganize metadata domains into classification/ and detection/ subdirs
- add annotation_path to metadata and split wrapper by task type

### Fixed

- guard show_samples_for_dataset for detection tasks
- detection config warnings and eval adapter cleanup
- validate monitor_metric for detection tasks
- reject timm/ architectures for detection tasks
- task-aware training curves label and use_tta detection warning
- warn on ignored classification params for detection tasks

### Miscellaneous

- suppress Sonar S1172 for detection adapter protocol params
- detection cosmetic fixes and missing tests

## [0.2.2] - 2026-03-18

### Added

- add dump_git_info to audit pipeline and harden mutation scores
- add task abstraction layer with strategy dispatch
- add TaskTrainingStep protocol for task-agnostic training forward

### Build

- bump black in the uv group across 1 directory (#3)
- bump pyasn1 in the uv group across 1 directory (#4)

### CI

- remove freshness check from publish workflow

### Changed

- generalize task eval protocols to Mapping[str, float]
- inject TaskValidationMetrics into training loop and Optuna executor
- add strict type annotations across all test files
- unify monitor_metric direction as SSOT in TrainingConfig
- generalize TrainingReport + fix dump_requirements for uv
- skip _check_min_dataset_size for non-classification tasks
- skip compute_class_weights for non-classification tasks
- derive fallback_metrics from task registry
- derive early-stopping thresholds from task registry
- pre-release audit — task-agnostic fixes, constant consolidation, mutant killers

### Documentation

- fix notebook setup cells and add edge deploy notebook
- task-agnostic docstring cleanup + package structure update

### Fixed

- use --no-deps to avoid reinstalling Colab dependencies
- ensure clean orchard-ml install on Colab re-runs
- fix API calls, suppress HF progress bar, clean up configs
- suppress HF download noise and fix notebook 03 Colab errors
- notebook 03 Optuna best params and remove quantize from pipeline
- add notebook 03 to README and update stale comments
- cleanup — centralize HIGHRES_THRESHOLD, pathlib consistency, stale docs
- remove redundant FileNotFoundError catch in serialization
- remove requirements.txt, use pyproject.toml as single source of truth
- harden evaluation pipeline, param mapping, and config safety
- bump black>=26.3.1 (CVE-2026-32274) and dismiss onnx hub.load alert

## [0.2.1] - 2026-03-09

### Added

- add --set override safety guards for Optuna and export configs
- centralize mutmut log/warn skip, add mutation registry guards

### Changed

- use domain-specific exceptions instead of bare ValueError
- replace type: ignore with cast(), add CLI --set validation, improve tests
- move diagnostic utilities into data_handler/diagnostic submodule

### Documentation

- add mutmut badge and update mutation testing guide

### Fixed

- resolve CodeQL security warnings for unused variable and mixed returns
- merge implicitly concatenated f-strings in cli_app.py
- remove commented-out code flagged by SonarCloud (python:S125)
- escalate log levels in InfrastructureManager and improve mutation score to 100%
- handle Z suffix in isoformat dates for Python 3.10 compat
- add fetch-depth: 0 to publish checkout for freshness check

### Miscellaneous

- add .gitattributes to make YAML/Shell/Dockerfile detectable by Linguist
- refresh mutmut-registry timestamps for v0.2.1 release

## [0.2.0] - 2026-03-04

### Added

- enable mypy --strict across codebase, CI, and badge workflows
- add CodeQL SAST workflow, uv.lock for reproducible builds
- generate annotated YAML recipes with inline field documentation
- add strict warn_only mode, MPS determinism warning, and DDP-ready local_rank

### Changed

- replace f-string logger calls with lazy % formatting
- extract helpers from cli_app to reduce cyclomatic complexity
- introduce AuditSaverProtocol to reduce RootOrchestrator params

### Documentation

- add SECURITY.md with disclosure policy and supported versions

### Fixed

- use npt.NDArray[Any] for cross-version mypy --strict compatibility
- resolve all CodeQL code scanning alerts in test suite
- resolve CodeQL alert #13 — remove dual import of orchard.core.paths
- harden RootOrchestrator lifecycle guards and error handling
- merge implicit string concatenations in logger calls
- harden RootOrchestrator phase 6 fail-fast and fix mypy callable inference
- harden quantization, Optuna reporting, and fix misleading docs

### Miscellaneous

- sync uv.lock for v0.2.0

## [0.1.9] - 2026-03-03

### Added

- add multi-format quantization (INT8/UINT8/INT4/UINT4)

### Changed

- centralise device placement, add resolution validator, cleanup naming
- centralise device placement, add resolution validator, cleanup naming
- introduce custom exception hierarchy and unify MPS accelerator support
- dependency bounds, CI supply chain, Dockerfile rootless, config validators
- wire unwired config fields, fix search spaces, harden CI
- replace bare trial dicts with TrialData frozen dataclass
- enforce runtime immutability with MappingProxyType, disambiguate export logs
- harden types, fix NaN guard, add missing test markers

### Documentation

- fix inaccuracies across README and guide files

### Fixed

- merge implicit f-string concatenation in EvaluationConfig validator
- restore per-epoch trainer logging and early-stop summary
- remove duplicated branch in transforms, tidy README badges
- close leaked fd in synthetic.py, align bandit CI config, update docs
- update Docker badge to CUDA 12.6, remove stale notebook config fields

## [0.1.8] - 2026-02-28

### Added

- add f1 as monitor_metric, compute F1 in validate_epoch, drop unused deps
- add 128×128 resolution support, convert medical registry to YAML
- harden training pipeline — AUC NaN fallback, subsampling guard, best_val_f1, +25 tests
- Optuna loss search space + pre-release audit fixes

### Changed

- unify scheduler stepping on monitor_metric
- extract LoopOptions dataclass, fix docstrings in trainer/
- unify metric source — remove optuna.metric_name, use training.monitor_metric as SSOT
- config narrowing — decouple builders, transforms, TTA and visualization from Config
- unify VisionDataset — eliminate LazyNPZDataset, add lazy mmap loading
- relocate LogStyle to paths.constants — break circular imports, unify log symbols
- config narrowing — decouple factories and loaders from Config
- config narrowing + mypy warn_return_any + quality scripts

### Fixed

- false docstrings, dead field, sentinel logging, LogStyle.FAILURE
- remove unused `classes` param, reduce ColorFormatter complexity
- reduce run_final_evaluation params from 14 to 13 (Sonar S107)

### Styling

- uniformize docstrings across orchard/, enable ruff D rules

## [0.1.7] - 2026-02-25

### Added

- ONNX INT8 quantization + MkDocs API documentation
- add configurable TTA blur kernel size + fix docs 404

### Build

- replace flake8/isort with ruff, enforce 100% coverage, add pre-push hook

### Changed

- extract TrainingLoop to unify trainer and executor epoch logic
- remove dead code and harden config immutability
- unify log styling with LogStyle across entire pipeline
- narrow exception types and reduce interrupt grace period

### Documentation

- fix all mkdocs/griffe warnings and improve API docstrings
- add usage section to FRAMEWORK.md, fix Quick Start in index.md
- improve module-level docstrings for trainer, export, and evaluation
- dynamic test count badges, fix HTML rendering and changelog entry

### Fixed

- mkdocs build warnings — remove --strict, fix broken links
- fix:
- resolve SonarQube code smells in logger and export phase

### Miscellaneous

- fix test typo, add MIT classifier, document CI decisions

## [0.1.6] - 2026-02-23

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
- audit hardening — TrackingConfig frozen, Bandit skips, manifest dedup

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
