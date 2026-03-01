"""
Experiment Lifecycle Orchestration.

This module provides RootOrchestrator, the central coordinator for experiment
execution. It manages the complete lifecycle from configuration validation to
resource cleanup, ensuring deterministic and reproducible ML experiments.

Architecture:

- Dependency Injection: All external dependencies are injectable for testability
- 7-Phase Initialization: Sequential setup from seeding to environment reporting
- Context Manager: Automatic resource acquisition and cleanup
- Protocol-Based: type-safe abstractions for mockability

Key Components:

- ``RootOrchestrator``: Main lifecycle controller

Related Protocols (defined in their respective modules):

- ``InfraManagerProtocol``: ``config/infrastructure_config.py``
- ``ReporterProtocol``: ``logger/reporter.py``
- ``TimeTrackerProtocol``: ``environment/timing.py``

Example:
    >>> from orchard.core import Config, RootOrchestrator
    >>> cfg = Config.from_recipe(Path("recipes/config_mini_cnn.yaml"))
    >>> with RootOrchestrator(cfg) as orchestrator:
    ...     device = orchestrator.get_device()
    ...     paths = orchestrator.paths
    ...     # Run training pipeline phases
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Literal, TypeVar

import torch

from .config.infrastructure_config import InfraManagerProtocol, InfrastructureManager
from .environment import (
    apply_cpu_threads,
    configure_system_libraries,
    get_rank,
    set_seed,
    to_device_obj,
)
from .environment.timing import TimeTracker, TimeTrackerProtocol
from .io import dump_requirements, save_config_as_yaml
from .logger import Logger, Reporter
from .logger.reporter import ReporterProtocol
from .paths import LOGGER_NAME, RunPaths, setup_static_directories

if TYPE_CHECKING:  # pragma: no cover
    from .config.manifest import Config

logger = logging.getLogger(LOGGER_NAME)

T = TypeVar("T")


def _resolve(value: T | None, default_factory: Callable[[], T]) -> T:
    """
    Resolve optional dependency with lazy default instantiation.

    Centralizes the None-check pattern used across RootOrchestrator.__init__
    for dependency injection. When the caller provides an explicit value, it
    is returned as-is; otherwise the default_factory is invoked to create
    the production default. This avoids mutable default arguments and defers
    heavy object construction until actually needed.

    Args:
        value: Caller-supplied dependency, or None to use the default.
        default_factory: Zero-argument callable that produces the default
            instance (e.g., ``InfrastructureManager``, ``Reporter``).

    Returns:
        The resolved dependency — either the provided value or a fresh default.
    """
    return value if value is not None else default_factory()


# ROOT ORCHESTRATOR
class RootOrchestrator:
    """
    Central coordinator for ML experiment lifecycle management.

    Orchestrates the complete initialization sequence from configuration validation
    through resource provisioning to execution readiness. Implements a 7-phase
    initialization protocol with dependency injection for maximum testability.

    The orchestrator follows the Single Responsibility Principle by delegating
    specialized tasks to injected dependencies while maintaining overall coordination.
    Uses the Context Manager pattern to guarantee resource cleanup even during failures.

    Initialization Phases:

    1. Determinism: Global RNG seeding (Python, NumPy, PyTorch)
    2. Runtime Configuration: CPU thread affinity, system libraries
    3. Filesystem Provisioning: Dynamic workspace creation via RunPaths
    4. Logging Initialization: File-based persistent logging setup
    5. Config Persistence: YAML manifest export for auditability
    6. Infrastructure Guarding: OS-level resource locks (prevents race conditions)
    7. Environment Reporting: Comprehensive telemetry logging

    Dependency Injection:

    All external dependencies are injectable with sensible defaults:

    - infra_manager: OS resource management (locks, cleanup)
    - reporter: Environment telemetry engine
    - log_initializer: Logging setup strategy
    - seed_setter: RNG seeding function
    - thread_applier: CPU thread configuration
    - system_configurator: System library setup (matplotlib, etc)
    - static_dir_setup: Static directory creation
    - config_saver: YAML persistence function
    - device_resolver: Hardware device detection

    Attributes:
        cfg (Config): Validated global configuration (Single Source of Truth)
        rank (int): Global rank of this process (0 in single-process mode)
        is_main_process (bool): True for rank 0, False for non-main ranks
        infra (InfraManagerProtocol): Infrastructure resource manager
        reporter (ReporterProtocol): Environment telemetry engine
        time_tracker (TimeTrackerProtocol): Pipeline duration tracker
        paths (RunPaths | None): Session-specific directory structure (None on non-main ranks)
        run_logger (logging.Logger | None): Active logger instance (None on non-main ranks)
        repro_mode (bool): Strict determinism flag
        num_workers (int): DataLoader worker processes

    Example:
        >>> cfg = Config.from_recipe(Path("recipes/config_mini_cnn.yaml"))
        >>> with RootOrchestrator(cfg) as orch:
        ...     device = orch.get_device()
        ...     logger = orch.run_logger
        ...     paths = orch.paths
        ...     # Execute training pipeline with guaranteed cleanup

    Notes:

    - Thread-safe: Single-instance locking via InfrastructureManager
    - Idempotent: initialize_core_services() is safe to call multiple times
      (subsequent calls return cached RunPaths without re-executing phases)
    - Auditable: All configuration saved to YAML in workspace
    - Deterministic: Reproducible experiments via strict seeding
    """

    def __init__(
        self,
        cfg: "Config",
        infra_manager: InfraManagerProtocol | None = None,
        reporter: ReporterProtocol | None = None,
        time_tracker: TimeTrackerProtocol | None = None,
        log_initializer: Callable | None = None,
        seed_setter: Callable | None = None,
        thread_applier: Callable | None = None,
        system_configurator: Callable | None = None,
        static_dir_setup: Callable | None = None,
        config_saver: Callable | None = None,
        requirements_dumper: Callable | None = None,
        device_resolver: Callable | None = None,
        rank: int | None = None,
    ) -> None:
        """
        Initializes orchestrator with dependency injection.

        Args:
            cfg: Validated global configuration (SSOT)
            infra_manager: Infrastructure management handler (default: InfrastructureManager())
            reporter: Environment reporting engine (default: Reporter())
            time_tracker: Pipeline duration tracker (default: TimeTracker())
            log_initializer: Logging setup function (default: Logger.setup)
            seed_setter: RNG seeding function (default: set_seed)
            thread_applier: CPU thread configuration (default: apply_cpu_threads)
            system_configurator: System library setup (default: configure_system_libraries)
            static_dir_setup: Static directory creation (default: setup_static_directories)
            config_saver: Config persistence (default: save_config_as_yaml)
            requirements_dumper: Dependency snapshot (default: dump_requirements)
            device_resolver: Device resolution (default: to_device_obj)
            rank: Global rank of this process (default: auto-detected from RANK env var).
                Rank 0 executes all phases; rank N skips filesystem, logging,
                config persistence, infrastructure locking, and reporting.
        """
        self.cfg = cfg

        # Dependency injection with defaults (using _resolve for uniform None-handling)
        self.rank = _resolve(rank, get_rank)
        self.is_main_process = self.rank == 0
        self.infra = _resolve(infra_manager, InfrastructureManager)
        self.reporter = _resolve(reporter, Reporter)
        self.time_tracker = _resolve(time_tracker, TimeTracker)
        self._log_initializer = log_initializer or Logger.setup
        self._seed_setter = seed_setter or set_seed
        self._thread_applier = thread_applier or apply_cpu_threads
        self._system_configurator = system_configurator or configure_system_libraries
        self._static_dir_setup = static_dir_setup or setup_static_directories
        self._config_saver = config_saver or save_config_as_yaml
        self._requirements_dumper = requirements_dumper or dump_requirements
        self._device_resolver = device_resolver or to_device_obj

        # Lazy initialization
        self._initialized: bool = False
        self._applied_threads: int = 0
        self.paths: RunPaths | None = None
        self.run_logger: logging.Logger | None = None
        self._device_cache: torch.device | None = None

        # Policy extraction from SSOT
        self.repro_mode = self.cfg.hardware.use_deterministic_algorithms
        self.num_workers = self.cfg.hardware.effective_num_workers

    def __enter__(self) -> "RootOrchestrator":
        """
        Context Manager entry — triggers the 7-phase initialization sequence.

        Starts the pipeline timer and delegates to initialize_core_services()
        for deterministic seeding, filesystem provisioning, logging setup,
        config persistence, infrastructure locking, and environment reporting.

        If any phase raises, cleanup() is called before re-raising to ensure
        partial resources (locks, file handles) are released even on failure.

        Returns:
            Fully initialized RootOrchestrator ready for pipeline execution.

        Raises:
            Exception: Re-raises any initialization error after cleanup.
        """
        try:
            self.time_tracker.start()
            self.initialize_core_services()
            return self
        except Exception:
            self.cleanup()
            raise

    def __exit__(self, exc_type, exc_val, exc_tb) -> Literal[False]:
        """
        Context Manager exit — logs duration and guarantees resource teardown.

        Invoked automatically when leaving the ``with`` block, whether the
        pipeline completed normally or raised an exception. Stops the timer,
        emits the total pipeline duration to the active logger, then delegates
        to cleanup() for infrastructure lock release and logging handler closure.

        Returns False so that any exception propagates to the caller unchanged;
        The CLI's top-level handler is responsible for user-facing error reporting.

        Args:
            exc_type (type[BaseException] | None): Exception class if the block raised, else None.
            exc_val (BaseException | None): Exception instance if the block raised, else None.
            exc_tb (TracebackType | None): Traceback object if the block raised, else None.

        Returns:
            Always False — exceptions are never suppressed.
        """
        # Stop timer (duration already shown in pipeline summary)
        self.time_tracker.stop()

        self.cleanup()
        return False

    # --- Private Lifecycle Phases ---
    # Each phase emits a logger.debug marker for diagnostics.
    # To activate: set log_level: DEBUG in the telemetry YAML section.

    def _phase_1_determinism(self) -> None:
        """Enforces global RNG seeding and algorithmic determinism."""
        logger.debug(  # pragma: no mutant
            "Phase 1: Applying deterministic seeding (seed=%d)", self.cfg.training.seed
        )
        self._seed_setter(self.cfg.training.seed, strict=self.repro_mode)

    def _phase_2_runtime_configuration(self) -> int:
        """
        Configures compute thread affinity and system libraries.

        Returns:
            Number of CPU threads applied to runtime
        """
        logger.debug(  # pragma: no mutant
            "Phase 2: Configuring runtime (workers=%d)", self.num_workers
        )
        applied_threads = self._thread_applier(self.num_workers)
        self._system_configurator()
        return applied_threads  # type: ignore[no-any-return]

    def _phase_3_filesystem_provisioning(self) -> None:
        """
        Constructs experiment workspace via RunPaths.

        Anchors relative paths to validated PROJECT_ROOT.
        """
        logger.debug("Phase 3: Provisioning filesystem")  # pragma: no mutant
        self._static_dir_setup()
        self.paths = RunPaths.create(
            dataset_slug=self.cfg.dataset.dataset_name,
            architecture_name=self.cfg.architecture.name,
            training_cfg=self.cfg.dump_serialized(),
            base_dir=self.cfg.telemetry.output_dir,
        )

    def _phase_4_logging_initialization(self) -> None:
        """
        Bridges static Logger to session-specific filesystem.

        Reconfigures handlers for file-based persistence in run directory.
        """
        logger.debug("Phase 4: Initializing session logging")  # pragma: no mutant
        assert self.paths is not None, "Paths must be initialized before logging"  # nosec B101
        self.run_logger = self._log_initializer(
            name=LOGGER_NAME, log_dir=self.paths.logs, level=self.cfg.telemetry.log_level
        )

    def _phase_5_run_manifest(self) -> None:
        """
        Persists run manifest: config YAML and frozen dependency snapshot.

        Ensures full reproducibility from artifacts alone.
        """
        logger.debug(  # pragma: no mutant
            "Phase 5: Persisting run manifest (config + requirements)"
        )
        assert (
            self.paths is not None
        ), "Paths must be initialized before config persistence"  # nosec B101
        self._config_saver(data=self.cfg, yaml_path=self.paths.get_config_path())
        self._requirements_dumper(self.paths.reports / "requirements.txt")

    def _phase_6_infrastructure_guarding(self) -> None:
        """
        Secures system-level resource locks via InfrastructureManager.

        Prevents concurrent execution conflicts and manages cleanup.
        """
        logger.debug("Phase 6: Acquiring infrastructure locks")  # pragma: no mutant
        phase_logger = self.run_logger or logging.getLogger(LOGGER_NAME)
        if self.infra is not None:
            try:
                self.infra.prepare_environment(self.cfg, logger=phase_logger)
            except (OSError, RuntimeError) as e:
                phase_logger.warning(f"Infra guard failed: {e}")

    def _phase_7_environment_report(self, applied_threads: int) -> None:
        """
        Emits baseline environment report to active logging streams.

        Summarizes hardware, dataset metadata, and execution policies.
        """
        logger.debug("Phase 7: Generating environment report")  # pragma: no mutant
        assert self.paths is not None, "Paths must be initialized before reporting"  # nosec B101
        phase_logger = self.run_logger or logging.getLogger(LOGGER_NAME)

        if self._device_cache is None:
            try:
                self._device_cache = self.get_device()
            except RuntimeError as e:
                # Last-resort safety net: resolve_device in HardwareConfig already
                # warns at config-time when GPU is unavailable. This catch handles
                # the unlikely case where to_device_obj() fails at runtime despite
                # a valid config (e.g. driver crash after config was built).
                self._device_cache = torch.device("cpu")
                phase_logger.warning(f"Device detection failed, fallback to CPU: {e}")

        self.reporter.log_initial_status(
            logger_instance=phase_logger,
            cfg=self.cfg,
            paths=self.paths,
            device=self._device_cache,
            applied_threads=applied_threads,
            num_workers=self.num_workers,
        )

    def _close_logging_handlers(self) -> None:
        """
        Flush and close all logging handlers to release file resources.

        This is distinct from Logger._setup_logger() handler cleanup, which serves
        a different purpose in the lifecycle:

        - Logger._setup_logger(): Removes OLD handlers during RECONFIGURATION
          (e.g., when transitioning from console-only to console+file logging).
          This is an idempotency guard that prevents duplicate handlers when the
          logger is re-initialized with a new log_dir.

        - This method (_close_logging_handlers): Releases ACTIVE handlers at
          PIPELINE END. Called by cleanup() when the RootOrchestrator context
          manager exits, ensuring RotatingFileHandler file locks are released
          and buffered log data is flushed to disk. Without this, log files
          may remain locked or lose trailing entries on abrupt shutdown.

        Both are necessary: _setup_logger() ensures clean transitions between
        logging phases, while this method ensures clean resource release at exit.
        """
        if self.run_logger:
            for handler in self.run_logger.handlers[:]:
                handler.close()
                self.run_logger.removeHandler(handler)

    # --- Public Interface ---

    def initialize_core_services(self) -> RunPaths | None:
        """
        Executes linear sequence of environment initialization phases.

        Synchronizes global state through 7 phases, progressing from
        deterministic seeding to full environment reporting.

        In distributed mode (torchrun / DDP), only the main process (rank 0)
        executes phases 3-7 (filesystem, logging, config persistence, infra
        locking, reporting).  All ranks execute phases 1-2 (seeding, threads)
        to ensure identical RNG state and thread affinity.

        Idempotent: guarded by ``_initialized`` flag. If already initialized,
        returns existing RunPaths without re-executing any phase. This prevents
        orphaned directories (Phase 3 creates unique paths per call) and
        resource leaks (Phase 6 acquires filesystem locks).

        Returns:
            Provisioned directory structure for rank 0, None for non-main ranks.
        """
        if self._initialized:
            return self.paths

        # All ranks: deterministic seeding and thread configuration
        self._phase_1_determinism()
        applied_threads = self._phase_2_runtime_configuration()

        # Rank 0 only: filesystem, logging, persistence, locking, reporting
        if self.is_main_process:
            self._phase_3_filesystem_provisioning()
            self._phase_4_logging_initialization()

            # type guards: paths and logger are guaranteed after phases 3-4
            assert self.paths is not None, "Paths not initialized after phase 3"  # nosec B101
            assert self.run_logger is not None, "Logger not initialized after phase 4"  # nosec B101

            self._phase_5_run_manifest()
            self._phase_6_infrastructure_guarding()
            self._applied_threads = applied_threads
        else:
            self._applied_threads = applied_threads
            logger.debug(  # pragma: no mutant
                "Rank %d: skipping phases 3-7 (non-main process).", self.rank
            )

        self._initialized = True
        return self.paths

    def log_environment_report(self) -> None:
        """
        Emit the environment initialization report (phase 7).

        Designed to be called explicitly by the CLI after external services
        (e.g. MLflow tracker) have been started, so that all enter/exit log
        messages appear in the correct chronological order.
        """
        if self.is_main_process:
            self._phase_7_environment_report(self._applied_threads)

    def cleanup(self) -> None:
        """
        Releases system resources and removes execution lock file.

        Guarantees clean state for subsequent runs by unlinking
        InfrastructureManager guards and closing logging handlers.
        Non-main ranks skip resource release (they never acquired locks
        or opened file-based log handlers).
        """
        if not self.is_main_process:
            return

        cleanup_logger = self.run_logger or logging.getLogger(LOGGER_NAME)
        try:
            if self.infra:
                self.infra.release_resources(self.cfg, logger=cleanup_logger)
        except (OSError, RuntimeError) as e:
            cleanup_logger.error(f"Failed to release system lock: {e}")

        self._close_logging_handlers()

    def get_device(self) -> torch.device:
        """
        Resolves and caches optimal computation device (CUDA/CPU/MPS).

        Returns:
            PyTorch device object for model execution
        """
        if self._device_cache is None:
            self._device_cache = self._device_resolver(device_str=self.cfg.hardware.device)
        return self._device_cache
