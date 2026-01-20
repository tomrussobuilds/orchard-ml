"""
Experiment Lifecycle Orchestration.

This module provides RootOrchestrator, the central coordinator for experiment
execution. It manages the complete lifecycle from configuration validation to
resource cleanup, ensuring deterministic and reproducible ML experiments.

Architecture:
    - Dependency Injection: All external dependencies are injectable for testability
    - 7-Phase Initialization: Sequential setup from seeding to environment reporting
    - Context Manager: Automatic resource acquisition and cleanup
    - Protocol-Based: Type-safe abstractions for mockability

Key Components:
    RootOrchestrator: Main lifecycle controller
    InfraManagerProtocol: Abstract interface for infrastructure management
    ReporterProtocol: Abstract interface for environment telemetry

Typical Usage:
    >>> from orchard.core import Config, RootOrchestrator
    >>> cfg = Config.from_yaml("config.yaml")
    >>> with RootOrchestrator(cfg) as orchestrator:
    ...     device = orchestrator.get_device()
    ...     paths = orchestrator.paths
    ...     # Run training pipeline
"""

# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
import logging
from typing import TYPE_CHECKING, Callable, Optional, Protocol

# =========================================================================== #
#                             Third-Party Imports                             #
# =========================================================================== #
import torch

# =========================================================================== #
#                                Internal Imports                             #
# =========================================================================== #
from .environment import (
    apply_cpu_threads,
    configure_system_libraries,
    set_seed,
    to_device_obj,
)
from .io import save_config_as_yaml
from .logger import Logger, Reporter
from .paths import LOGGER_NAME, RunPaths, setup_static_directories
from .config.infrastructure_config import InfrastructureManager

if TYPE_CHECKING:
    from .config.engine import Config


# =========================================================================== #
#                            Protocols (for DI)                               #
# =========================================================================== #


class InfraManagerProtocol(Protocol):
    """Protocol for infrastructure management (allows mocking)."""

    def prepare_environment(self, cfg: "Config", logger: logging.Logger) -> None:
        """Prepare system resources."""
        ...

    def release_resources(self, cfg: "Config", logger: logging.Logger) -> None:
        """Release system resources."""
        ...


class ReporterProtocol(Protocol):
    """Protocol for environment reporting (allows mocking)."""

    def log_initial_status(
        self,
        logger_instance: logging.Logger,
        cfg: "Config",
        paths: RunPaths,
        device: torch.device,
        applied_threads: int,
        num_workers: int,
    ) -> None:
        """Log environment initialization status."""
        ...


# =========================================================================== #
#                            Root Orchestrator                                #
# =========================================================================== #


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
        2. Hardware Optimization: CPU thread configuration, system libraries
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
        infra (InfraManagerProtocol): Infrastructure resource manager
        reporter (ReporterProtocol): Environment telemetry engine
        paths (RunPaths): Session-specific directory structure
        run_logger (logging.Logger): Active logger instance for session
        repro_mode (bool): Strict determinism flag
        num_workers (int): DataLoader worker processes

    Example:
        >>> cfg = Config.from_args(args)
        >>> with RootOrchestrator(cfg) as orch:
        ...     device = orch.get_device()
        ...     logger = orch.run_logger
        ...     paths = orch.paths
        ...     # Execute training pipeline with guaranteed cleanup

    Notes:
        - Thread-safe: Single-instance locking via InfrastructureManager
        - Idempotent: Multiple initialization attempts are safe
        - Auditable: All configuration saved to YAML in workspace
        - Deterministic: Reproducible experiments via strict seeding
    """

    def __init__(
        self,
        cfg: "Config",
        infra_manager: Optional[InfraManagerProtocol] = None,
        reporter: Optional[ReporterProtocol] = None,
        log_initializer: Optional[Callable] = None,
        seed_setter: Optional[Callable] = None,
        thread_applier: Optional[Callable] = None,
        system_configurator: Optional[Callable] = None,
        static_dir_setup: Optional[Callable] = None,
        config_saver: Optional[Callable] = None,
        device_resolver: Optional[Callable] = None,
    ) -> None:
        """
        Initializes orchestrator with dependency injection.

        Args:
            cfg: Validated global configuration (SSOT)
            infra_manager: Infrastructure management handler (default: InfrastructureManager())
            reporter: Environment reporting engine (default: Reporter())
            log_initializer: Logging setup function (default: Logger.setup)
            seed_setter: RNG seeding function (default: set_seed)
            thread_applier: CPU thread configuration (default: apply_cpu_threads)
            system_configurator: System library setup (default: configure_system_libraries)
            static_dir_setup: Static directory creation (default: setup_static_directories)
            config_saver: Config persistence (default: save_config_as_yaml)
            device_resolver: Device resolution (default: to_device_obj)
        """
        self.cfg = cfg

        # Dependency injection with defaults

        self.infra = infra_manager if infra_manager is not None else InfrastructureManager()
        self.reporter = reporter or Reporter()
        self._log_initializer = log_initializer or Logger.setup
        self._seed_setter = seed_setter or set_seed
        self._thread_applier = thread_applier or apply_cpu_threads
        self._system_configurator = system_configurator or configure_system_libraries
        self._static_dir_setup = static_dir_setup or setup_static_directories
        self._config_saver = config_saver or save_config_as_yaml
        self._device_resolver = device_resolver or to_device_obj

        # Lazy initialization
        self.paths: Optional[RunPaths] = None
        self.run_logger: Optional[logging.Logger] = None
        self._device_cache: Optional[torch.device] = None

        # Policy extraction from SSOT
        self.repro_mode = self.cfg.hardware.use_deterministic_algorithms
        self.num_workers = self.cfg.hardware.effective_num_workers

    def __enter__(self) -> "RootOrchestrator":
        """
        Context Manager entry - triggers core service initialization.

        Returns:
            Initialized RootOrchestrator ready for pipeline execution
        """
        try:
            self.initialize_core_services()
            return self
        except Exception as e:
            self.cleanup()
            raise e

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """
        Context Manager exit - ensures resource release and lock cleanup.

        Args:
            exc_type: Exception type if raised
            exc_val: Exception instance if raised
            exc_tb: Exception traceback if raised

        Returns:
            False to allow exception propagation
        """
        self.cleanup()
        return False

    # --- Private Lifecycle Phases ---

    def _phase_1_determinism(self) -> None:
        """Enforces global RNG seeding and algorithmic determinism."""
        self._seed_setter(self.cfg.training.seed, strict=self.repro_mode)

    def _phase_2_hardware_optimization(self) -> int:
        """
        Configures compute thread affinity and accelerator libraries.

        Returns:
            Number of CPU threads applied to runtime
        """
        applied_threads = self._thread_applier(self.num_workers)
        self._system_configurator()
        return applied_threads

    def _phase_3_filesystem_provisioning(self) -> None:
        """
        Constructs experiment workspace via RunPaths.
        Anchors relative paths to validated PROJECT_ROOT.
        """
        self._static_dir_setup()
        self.paths = RunPaths.create(
            dataset_slug=self.cfg.dataset.dataset_name,
            model_name=self.cfg.model.name,
            training_cfg=self.cfg.dump_serialized(),
            base_dir=self.cfg.telemetry.output_dir,
        )

    def _phase_4_logging_initialization(self) -> None:
        """
        Bridges static Logger to session-specific filesystem.
        Reconfigures handlers for file-based persistence in run directory.
        """
        self.run_logger = self._log_initializer(
            name=LOGGER_NAME, log_dir=self.paths.logs, level=self.cfg.telemetry.log_level
        )

    def _phase_5_config_persistence(self) -> None:
        """
        Mirrors hydrated configuration to experiment root.
        Saves portable YAML manifest for session auditability.
        """
        self._config_saver(data=self.cfg, yaml_path=self.paths.get_config_path())

    def _phase_6_infrastructure_guarding(self) -> None:
        """
        Secures system-level resource locks via InfrastructureManager.
        Prevents concurrent execution conflicts and manages cleanup.
        """
        if self.infra:
            self.infra.prepare_environment(self.cfg, logger=self.run_logger)

    def _phase_7_environment_reporting(self, applied_threads: int) -> None:
        """
        Emits baseline environment report to active logging streams.
        Summarizes hardware, dataset metadata, and execution policies.
        """
        self.reporter.log_initial_status(
            logger_instance=self.run_logger,
            cfg=self.cfg,
            paths=self.paths,
            device=self.get_device(),
            applied_threads=applied_threads,
            num_workers=self.num_workers,
        )

    # --- Public Interface ---

    def initialize_core_services(self) -> RunPaths:
        """
        Executes linear sequence of environment initialization phases.

        Synchronizes global state through 7 phases, progressing from
        deterministic seeding to full environment reporting.

        Returns:
            Verified and provisioned directory structure
        """
        self._phase_1_determinism()
        applied_threads = self._phase_2_hardware_optimization()
        self._phase_3_filesystem_provisioning()
        self._phase_4_logging_initialization()
        self._phase_5_config_persistence()
        self._phase_6_infrastructure_guarding()
        self._phase_7_environment_reporting(applied_threads)

        return self.paths

    def cleanup(self) -> None:
        """
        Releases system resources and removes execution lock file.

        Guarantees clean state for subsequent runs by unlinking
        InfrastructureManager guards and closing logging handlers.
        """
        try:
            if self.infra:
                self.infra.release_resources(self.cfg, logger=self.run_logger)
        except Exception as e:
            err_msg = f"Failed to release system lock: {e}"
            if self.run_logger:
                for handler in self.run_logger.handlers[:]:
                    handler.close()
                    self.run_logger.removeHandler(handler)
            else:
                logging.error(err_msg)

    def get_device(self) -> torch.device:
        """
        Resolves and caches optimal computation device (CUDA/CPU/MPS).

        Returns:
            PyTorch device object for model execution
        """
        if self._device_cache is None:
            self._device_cache = self._device_resolver(device_str=self.cfg.hardware.device)
        return self._device_cache
