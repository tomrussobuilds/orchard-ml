← [Back to Main README](../../README.md)

# Framework Architecture

## Core Features

### RootOrchestrator -- Lifecycle Management

The `RootOrchestrator` (`orchard/core/orchestrator.py`) is the central coordinator for every ML experiment. It implements a **7-phase initialization protocol** inside a Context Manager, guaranteeing deterministic setup and automatic resource cleanup:

| Phase | Responsibility | Key Detail |
|-------|---------------|------------|
| **1. Determinism** | Global RNG seeding | Python, NumPy, PyTorch (strict mode optional) |
| **2. Runtime Configuration** | CPU thread affinity, system libraries | matplotlib backend, library silencing |
| **3. Filesystem Provisioning** | Dynamic workspace via `RunPaths` | BLAKE2b-hashed directories |
| **4. Logging Initialization** | File-based persistent logging | Hot-swap from STDOUT to RotatingFileHandler |
| **5. Config Persistence** | YAML manifest export | Full audit trail in workspace |
| **6. Infrastructure Guarding** | OS-level resource locks (`flock`) | Prevents concurrent run collisions |
| **7. Environment Reporting** | Comprehensive telemetry | Hardware, dataset metadata, policies |

**Design qualities:**
- **Context Manager pattern**: `with RootOrchestrator(cfg) as orch:` guarantees cleanup even on failure -- lock release, handler flush, resource teardown
- **Full Dependency Injection**: Every external dependency (infra manager, reporter, seed setter, device resolver, ...) is injectable, enabling complete testability without side effects
- **Protocol-Based Abstractions**: `InfraManagerProtocol`, `ReporterProtocol`, `TimeTrackerProtocol` provide type-safe interfaces for mocking
- **Idempotent Initialization**: Guarded by `_initialized` flag -- safe to call multiple times without orphaned directories or lock leaks
- **Device Caching**: `get_device()` resolves and caches the optimal compute device (CUDA/CPU/MPS) once, avoiding repeated detection overhead

```python
args = parse_args()
cfg = Config.from_args(args)
with RootOrchestrator(cfg) as orch:
    device = orch.get_device()
    paths = orch.paths
    logger = orch.run_logger
    # Pipeline execution with guaranteed cleanup
```

### Configuration Engine (SSOT)

Built on Pydantic V2, the configuration system acts as a **Single Source of Truth**, transforming raw inputs (CLI/YAML) into an immutable, type-safe execution blueprint:

- **Late-Binding Metadata Injection**: Dataset specifications (normalization constants, class mappings) are resolved from a centralized registry at instantiation time
- **Cross-Domain Validation**: Post-construction logic guards prevent unstable states (e.g., enforcing RGB input for pretrained weights, validating AMP compatibility)
- **Path Portability**: Automatic serialization converts absolute paths to environment-agnostic anchors for cross-platform reproducibility

### Infrastructure Guard Layer

The `InfrastructureManager` bridges declarative configs with physical hardware (used by RootOrchestrator in Phase 6):

- **Mutual Exclusion via `flock`**: Kernel-level advisory locking ensures only one training instance per workspace (prevents VRAM race conditions)
- **Process Sanitization**: `psutil` wrapper identifies and terminates ghost Python processes
- **HPC-Aware Safety**: Auto-detects cluster schedulers (SLURM/PBS/LSF) and suspends aggressive process cleanup to preserve multi-user stability

### Reproducibility Architecture

**Dual-Layer Strategy** (enforced by RootOrchestrator Phase 1):
1. **Standard Mode**: Global seeding (Seed 42) with performance-optimized algorithms
2. **Strict Mode**: Bit-perfect reproducibility via:
   - `torch.use_deterministic_algorithms(True)`
   - `worker_init_fn` for multi-process RNG synchronization
   - Auto-scaling to `num_workers=0` when determinism is critical

**Data Integrity Validation:**
- MD5 checksum verification for dataset downloads
- `validate_npz_keys` structural integrity checks before memory allocation

### Performance Optimization

**Hybrid RAM Management:**
- **Small Datasets**: Full RAM caching for maximum throughput
- **Large Datasets**: Indexed slicing to prevent OOM errors

**Dynamic Path Anchoring:**
- "Search-up" logic locates project root via markers (`.git`, `README.md`)
- Ensures absolute path stability regardless of invocation directory

### Intelligent Hyperparameter Search

**Optuna Integration Features:**
- **TPE Sampling**: Tree-structured Parzen Estimator for efficient search space exploration
- **Median Pruning**: Early stopping of underperforming trials (30-50% time savings)
- **Persistent Studies**: SQLite storage enables resume-from-checkpoint
- **Type-Safe Constraints**: All search spaces respect Pydantic validation bounds
- **Auto-Visualization**: Parameter importance plots, optimization history, parallel coordinates

---

## System Architecture

The framework implements **Separation of Concerns (SoC)** with six core layers:

```
┌─────────────────────────────────────────────────────────────────┐
│                      RootOrchestrator                           │
│              (Lifecycle Manager & Context)                      │
│                                                                 │
│  Responsibilities:                                              │
│  • Phase 1-7 initialization sequence                            │
│  • Resource acquisition & cleanup (Context Manager)             │
│  • Device resolution & caching                                  │
└────────────┬─────────────────────────┬──────────────────────────┘
             │                         │
             │ uses                    │ uses
             │                         │
    ┌────────▼──────────┐     ┌────────▼───────────────┐
    │                   │     │                        │
    │  Config Engine    │     │  InfrastructureManager │
    │  (Pydantic V2)    │     │  (flock/psutil)        │
    │                   │     │                        │
    │  • Type safety    │     │  • Process cleanup     │
    │  • Validation     │     │  • Kernel locks        │
    │  • Metadata       │     │  • HPC detection       │
    │    injection      │     │                        │
    └───────────────────┘     └────────────────────────┘
             │
             │ provides config to
             │
    ┌────────▼───────────────────────────────────────────────┐
    │                                                        │
    │              Execution Pipeline                        │
    │                                                        │
    │  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
    │  │   Data   │  │  Model   │  │ Trainer  │              │
    │  │ Handler  │→ │ Factory  │→ │  Engine  │              │
    │  └──────────┘  └──────────┘  └────┬─────┘              │
    │                                   │                    │
    │                             ┌─────▼──────┐             │
    │                             │ Evaluation │             │
    │                             │  Pipeline  │             │
    │                             └────────────┘             │
    │                                                        │
    └──────────────────────────┬─────────────────────────────┘
                               │
            ┌──────────────────┼──────────────────┐
            │                  │                  │
            ▼ optional         ▼ alternative      ▼ alternative
    ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
    │ Tracking Engine  │  │  Optimization    │  │  Export Engine   │
    │    (MLflow)      │  │  Engine (Optuna) │  │ (ONNX/TScript)   │
    │                  │  │                  │  │                  │
    │ • Metric logging │  │ • Study mgmt     │  │ • Checkpoint load│
    │ • Param capture  │  │ • Trial exec     │  │ • ONNX convert   │
    │ • Run comparison │  │ • Pruning        │  │ • Validation     │
    │ • Local SQLite   │  │ • Visualization  │  │ • Benchmarking   │
    └──────────────────┘  └──────────────────┘  └──────────────────┘
```

**Key Design Principles:**

1. Orchestrator owns both Config and InfrastructureManager
2. Config is the SSOT - all modules receive it as dependency
3. InfrastructureManager is stateless utility for OS-level operations
4. Execution pipeline is linear: Data → Model → Training → Eval
5. Optimization wraps the entire pipeline for each trial
6. Export consumes saved checkpoints for production deployment
7. Tracking is opt-in and side-effect-free - disabled by default, logs to local SQLite via MLflow

---

## Dependency Graph

<p align="center">
<img src="../framework_map.svg?v=4" width="900" alt="System Dependency Graph">
</p>

> *Generated via `pydeps`. Highlights the centralized Config hub and modular architecture.*

**Regenerate:**
```bash
pydeps orchard \
    --cluster \
    --max-bacon=0 \
    --max-module-depth=5 \
    --only orchard \
    --noshow \
    -T svg \
    -o docs/framework_map.svg
```

> **Requirements:** `pydeps` + Graphviz (`sudo apt install graphviz` or `brew install graphviz`)

---
