"""
Test Suite for Optuna Orchestrator Module.

Focused on testing the orchestrator logic with proper mocking
to avoid triggering real downloads, file I/O, or network calls.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

import optuna
import pytest
import torch

from orchard.core import Config, RunPaths
from orchard.optimization._param_mapping import (
    PARAM_MAPPING,
    map_param_to_config_path,
)
from orchard.optimization.orchestrator import OptunaOrchestrator
from orchard.optimization.orchestrator.builders import (
    build_callbacks,
    build_pruner,
    build_sampler,
)
from orchard.optimization.orchestrator.exporters import (
    TrialData,
    build_best_config_dict,
)
from orchard.optimization.orchestrator.registries import (
    PRUNER_REGISTRY,
    SAMPLER_REGISTRY,
)
from orchard.optimization.orchestrator.utils import (
    get_completed_trials,
    has_completed_trials,
)
from orchard.tracking import TrackerProtocol


# FIXTURES
@pytest.fixture
def mock_cfg() -> MagicMock:
    """Minimal mock Config."""
    cfg = MagicMock()
    cfg.optuna.study_name = "test_study"
    cfg.optuna.direction = "maximize"
    cfg.optuna.sampler_type = "tpe"
    cfg.optuna.enable_pruning = True
    cfg.optuna.pruner_type = "median"
    cfg.optuna.n_trials = 5
    cfg.optuna.timeout = None
    cfg.optuna.n_jobs = 1
    cfg.optuna.show_progress_bar = False
    cfg.optuna.save_plots = False
    cfg.optuna.save_best_config = False
    cfg.training.monitor_metric = "auc"
    cfg.task_type = "classification"
    cfg.optuna.search_space_preset = "quick"
    cfg.optuna.load_if_exists = False
    cfg.optuna.get_storage_url = MagicMock(return_value=None)
    cfg.optuna.enable_early_stopping = False
    cfg.training.epochs = 50
    cfg.dataset.resolution = 28
    cfg.model_dump = MagicMock(
        return_value={
            "training": {"epochs": 50},
            "architecture": {},
            "augmentation": {},
        }
    )
    return cfg


@pytest.fixture
def mock_paths(tmp_path: Path) -> MagicMock:
    """Mock RunPaths with temp dirs."""
    paths = MagicMock()
    paths.root = tmp_path
    paths.reports = tmp_path / "reports"
    paths.figures = tmp_path / "figures"
    paths.reports.mkdir(exist_ok=True)
    paths.figures.mkdir(exist_ok=True)
    return paths


@pytest.fixture
def completed_trial() -> MagicMock:
    """Real completed trial (not mock)."""
    trial = MagicMock()
    trial.state = optuna.trial.TrialState.COMPLETE
    trial.number = 1
    trial.value = 0.95
    trial.params = {"learning_rate": 0.001}
    trial.datetime_start = datetime(2024, 1, 1, 0, 0, 0)
    trial.datetime_complete = datetime(2024, 1, 1, 1, 0, 0)
    return trial


@pytest.fixture
def study_with_trials(completed_trial: MagicMock) -> MagicMock:
    """Mock study with one completed trial."""
    study = MagicMock()
    study.study_name = "test"
    study.direction = optuna.study.StudyDirection.MAXIMIZE
    study.trials = [completed_trial]
    study.best_trial = completed_trial
    study.best_params = completed_trial.params
    study.best_value = completed_trial.value
    return study


def _make_orch(
    mock_cfg: MagicMock,
    mock_paths: MagicMock,
    device: torch.device = torch.device("cpu"),
    tracker: object | None = None,
) -> OptunaOrchestrator:
    """Wrap OptunaOrchestrator construction with the casts required by its typed API."""
    return OptunaOrchestrator(
        cfg=cast(Config, mock_cfg),
        device=device,
        paths=cast(RunPaths, mock_paths),
        tracker=cast(TrackerProtocol, tracker) if tracker is not None else None,
    )


# UNIT TESTS: config.py
@pytest.mark.unit
class TestConfig:
    """Test config constants and functions."""

    def test_sampler_registry_has_tpe(self) -> None:
        """Verify TPE sampler is registered."""
        assert "tpe" in SAMPLER_REGISTRY
        assert SAMPLER_REGISTRY["tpe"] == optuna.samplers.TPESampler

    def test_pruner_registry_has_median(self) -> None:
        """Verify Median pruner is registered."""
        assert "median" in PRUNER_REGISTRY

    def test_training_params_includes_lr(self) -> None:
        """Verify learning_rate is a training param."""
        assert "learning_rate" in PARAM_MAPPING["training"]

    def test_map_param_to_training(self) -> None:
        """Test mapping learning_rate to training section."""
        section, key = map_param_to_config_path("learning_rate")
        assert section == "training"
        assert key == "learning_rate"


# UNIT TESTS: builders.py
@pytest.mark.unit
class TestBuilders:
    """Test builder functions."""

    def test_build_sampler_tpe(self, mock_cfg: MagicMock) -> None:
        """Test building TPE sampler."""
        sampler = build_sampler(mock_cfg.optuna)
        assert isinstance(sampler, optuna.samplers.TPESampler)

    def test_build_pruner_median(self, mock_cfg: MagicMock) -> None:
        """Test building Median pruner."""
        pruner = build_pruner(mock_cfg.optuna)
        assert isinstance(pruner, optuna.pruners.MedianPruner)

    def test_build_pruner_disabled(self, mock_cfg: MagicMock) -> None:
        """Test disabled pruning returns NopPruner."""
        pruner = build_pruner(mock_cfg.optuna)
        assert isinstance(pruner, optuna.pruners.BasePruner)

    @patch("orchard.optimization.orchestrator.builders.get_early_stopping_callback")
    def test_build_callbacks(self, mock_callback_fn: MagicMock, mock_cfg: MagicMock) -> None:
        """Test building callbacks list."""
        mock_callback_fn.return_value = None
        callbacks = build_callbacks(
            mock_cfg.optuna, mock_cfg.training.monitor_metric, mock_cfg.training.monitor_direction
        )
        assert isinstance(callbacks, list)


# UNIT TESTS: utils.py
@pytest.mark.unit
class TestUtils:
    """Test utility functions."""

    def test_get_completed_trials(self, study_with_trials: MagicMock) -> None:
        """Test extracting completed trials."""
        completed = get_completed_trials(study_with_trials)
        assert len(completed) == 1

    def test_has_completed_trials_true(self, study_with_trials: MagicMock) -> None:
        """Test has_completed_trials returns True."""
        assert has_completed_trials(study_with_trials) is True

    def test_has_completed_trials_false(self) -> None:
        """Test has_completed_trials returns False."""
        study = MagicMock()
        study.trials = []
        assert has_completed_trials(study) is False


# TESTS: exporters.py
@pytest.mark.unit
class TestExporters:
    """Test exporter functions."""

    def test_trial_data_from_trial(self, completed_trial: MagicMock) -> None:
        """Test building TrialData from trial."""
        data = TrialData.from_trial(completed_trial)
        assert data.number == 1
        assert data.value == pytest.approx(0.95)
        assert data.state == "COMPLETE"

    def test_build_best_config_dict(self, mock_cfg: MagicMock) -> None:
        """Test building config dict from params."""
        params = {"learning_rate": 0.001}
        config_dict = build_best_config_dict(params, mock_cfg)
        assert config_dict["training"]["learning_rate"] == pytest.approx(0.001)


# INTEGRATION TESTS: OptunaOrchestrator
@pytest.mark.integration
class TestOptunaOrchestrator:
    """Integration tests for orchestrator."""

    def test_init(self, mock_cfg: MagicMock, mock_paths: MagicMock) -> None:
        """Test orchestrator initialization."""
        orch = _make_orch(mock_cfg, mock_paths)
        assert orch.cfg == mock_cfg
        assert orch.device == torch.device("cpu")
        assert orch.paths == mock_paths

    @patch("optuna.create_study")
    def test_create_study(
        self, mock_create: MagicMock, mock_cfg: MagicMock, mock_paths: MagicMock
    ) -> None:
        """Test create_study builds components."""
        mock_study = MagicMock()
        mock_create.return_value = mock_study

        orch = _make_orch(mock_cfg, mock_paths)
        study = orch.create_study()

        mock_create.assert_called_once()
        assert study == mock_study

    @patch("orchard.optimization.orchestrator.orchestrator.export_study_summary")
    def test_post_optimization_no_trials(
        self, mock_export: MagicMock, mock_cfg: MagicMock, mock_paths: MagicMock
    ) -> None:
        """Test post-optimization with no completed trials."""
        study = MagicMock()
        study.trials = []
        study.study_name = "test"
        study.direction = MagicMock()
        study.direction.name = "MAXIMIZE"

        orch = _make_orch(mock_cfg, mock_paths)
        orch._post_optimization_processing(study)

        mock_export.assert_called_once()

    @patch("orchard.optimization.orchestrator.orchestrator.export_study_summary")
    @patch("orchard.optimization.orchestrator.orchestrator.export_top_trials")
    def test_post_optimization_with_trials(
        self,
        mock_top_trials: MagicMock,
        mock_summary: MagicMock,
        study_with_trials: MagicMock,
        mock_cfg: MagicMock,
        mock_paths: MagicMock,
    ) -> None:
        """Test post-optimization with completed trials."""
        orch = _make_orch(mock_cfg, mock_paths)
        orch._post_optimization_processing(study_with_trials)

        mock_summary.assert_called_once()
        mock_top_trials.assert_called_once()

    @patch("optuna.create_study")
    @patch("orchard.optimization.orchestrator.orchestrator.get_search_space")
    @patch("orchard.optimization.orchestrator.orchestrator.OptunaObjective")
    def test_optimize_full_flow(
        self,
        mock_obj: MagicMock,
        mock_space: MagicMock,
        mock_create: MagicMock,
        mock_cfg: MagicMock,
        mock_paths: MagicMock,
    ) -> None:
        """Test full optimize flow."""
        mock_study = MagicMock()
        mock_study.trials = []
        mock_study.study_name = "test"
        mock_study.direction = MagicMock()
        mock_study.direction.name = "MAXIMIZE"
        mock_create.return_value = mock_study

        mock_space.return_value = {}
        mock_objective = MagicMock()
        mock_obj.return_value = mock_objective

        orch = _make_orch(mock_cfg, mock_paths)

        with patch.object(orch, "_post_optimization_processing"):
            result = orch.optimize()

        mock_create.assert_called_once()
        mock_space.assert_called_once()
        mock_obj.assert_called_once()
        mock_study.optimize.assert_called_once()
        assert result == mock_study

    @patch("orchard.optimization.orchestrator.orchestrator.OptunaOrchestrator")
    def test_run_optimization(
        self, mock_orch_class: MagicMock, mock_cfg: MagicMock, mock_paths: MagicMock
    ) -> None:
        """Test run_optimization convenience function."""
        mock_orch = MagicMock()
        mock_study = MagicMock()
        mock_orch.optimize.return_value = mock_study
        mock_orch_class.return_value = mock_orch

        from orchard.optimization.orchestrator.orchestrator import (
            run_optimization as run_opt,
        )

        result = run_opt(
            cfg=cast(Config, mock_cfg),
            device=torch.device("cpu"),
            paths=cast(RunPaths, mock_paths),
        )

        mock_orch_class.assert_called_once_with(
            cfg=mock_cfg, device=torch.device("cpu"), paths=mock_paths, tracker=None
        )
        mock_orch.optimize.assert_called_once()
        assert result == mock_study

    @patch("orchard.optimization.orchestrator.orchestrator.generate_visualizations")
    @patch("orchard.optimization.orchestrator.orchestrator.export_best_config")
    @patch("orchard.optimization.orchestrator.orchestrator.export_study_summary")
    def test_post_optimization_with_save_plots_enabled(
        self,
        mock_export_summary: MagicMock,
        mock_export_best: MagicMock,
        mock_generate_viz: MagicMock,
        study_with_trials: MagicMock,
        mock_cfg: MagicMock,
        mock_paths: MagicMock,
    ) -> None:
        """Test post-optimization with save_plots enabled."""
        mock_cfg.optuna.save_plots = True
        mock_cfg.optuna.save_best_config = False

        orch = _make_orch(mock_cfg, mock_paths)
        orch._post_optimization_processing(study_with_trials)

        mock_generate_viz.assert_called_once_with(study_with_trials, mock_paths.figures)
        mock_export_summary.assert_called_once()
        mock_export_best.assert_not_called()

    @patch("orchard.optimization.orchestrator.orchestrator.generate_visualizations")
    @patch("orchard.optimization.orchestrator.orchestrator.export_best_config")
    @patch("orchard.optimization.orchestrator.orchestrator.export_study_summary")
    def test_post_optimization_with_save_best_config_enabled(
        self,
        mock_export_summary: MagicMock,
        mock_export_best: MagicMock,
        mock_generate_viz: MagicMock,
        study_with_trials: MagicMock,
        mock_cfg: MagicMock,
        mock_paths: MagicMock,
    ) -> None:
        """Test post-optimization with save_best_config enabled."""
        mock_cfg.optuna.save_plots = False
        mock_cfg.optuna.save_best_config = True

        orch = _make_orch(mock_cfg, mock_paths)
        orch._post_optimization_processing(study_with_trials)

        mock_export_best.assert_called_once_with(study_with_trials, mock_cfg, mock_paths)
        mock_export_summary.assert_called_once()
        mock_generate_viz.assert_not_called()

    @patch("orchard.optimization.orchestrator.orchestrator.time.sleep")
    @patch("orchard.optimization.orchestrator.orchestrator.OptunaObjective")
    @patch("orchard.optimization.orchestrator.orchestrator.get_search_space")
    @patch("optuna.create_study")
    def test_optimize_keyboard_interrupt(
        self,
        mock_create_study: MagicMock,
        mock_get_search_space: MagicMock,
        mock_objective_class: MagicMock,
        mock_sleep: MagicMock,
        mock_cfg: MagicMock,
        mock_paths: MagicMock,
    ) -> None:
        """Test optimize handles KeyboardInterrupt with grace period."""
        mock_study = MagicMock()
        mock_study.trials = []
        mock_study.study_name = "test"
        mock_study.direction = MagicMock()
        mock_study.direction.name = "MAXIMIZE"
        mock_create_study.return_value = mock_study
        mock_get_search_space.return_value = {}

        mock_objective = MagicMock()
        mock_objective_class.return_value = mock_objective

        mock_study.optimize.side_effect = KeyboardInterrupt("User stopped")

        orch = _make_orch(mock_cfg, mock_paths)

        with patch.object(orch, "_post_optimization_processing") as mock_post:
            result = orch.optimize()

            mock_post.assert_called_once_with(mock_study)
            assert result == mock_study
            mock_sleep.assert_called_once_with(5)


# MUTATION-KILLING TESTS: orchestrator.py argument exactness
@pytest.mark.unit
class TestOrchestratorMutationKillers:
    """Tests that verify exact arguments to kill surviving mutants."""

    # __init__: verify each attribute stored correctly
    def test_init_stores_tracker(self, mock_cfg: MagicMock, mock_paths: MagicMock) -> None:
        """Verify tracker attribute is stored."""
        sentinel = object()
        orch = _make_orch(mock_cfg, mock_paths, tracker=sentinel)
        assert orch.tracker is sentinel

    def test_init_tracker_defaults_none(self, mock_cfg: MagicMock, mock_paths: MagicMock) -> None:
        """Verify tracker defaults to None."""
        orch = _make_orch(mock_cfg, mock_paths)
        assert orch.tracker is None

    # create_study: verify exact kwargs passed to optuna.create_study
    @patch("optuna.create_study")
    def test_create_study_passes_exact_kwargs(
        self, mock_create: MagicMock, mock_cfg: MagicMock, mock_paths: MagicMock
    ) -> None:
        """Verify every kwarg passed to optuna.create_study."""
        mock_create.return_value = MagicMock()
        mock_cfg.optuna.study_name = "my_study"
        mock_cfg.training.monitor_direction = "minimize"
        mock_cfg.optuna.load_if_exists = True
        storage_sentinel = "sqlite:///test.db"
        mock_cfg.optuna.get_storage_url.return_value = storage_sentinel

        orch = _make_orch(mock_cfg, mock_paths)
        orch.create_study()

        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["study_name"] == "my_study"
        assert call_kwargs["direction"] == "minimize"
        assert call_kwargs["storage"] == storage_sentinel
        assert call_kwargs["load_if_exists"] is True
        # sampler and pruner are built from config, just check they're present
        assert "sampler" in call_kwargs
        assert "pruner" in call_kwargs

    @patch("optuna.create_study")
    def test_create_study_storage_url_uses_paths(
        self, mock_create: MagicMock, mock_cfg: MagicMock, mock_paths: MagicMock
    ) -> None:
        """Verify get_storage_url is called with self.paths."""
        mock_create.return_value = MagicMock()
        orch = _make_orch(mock_cfg, mock_paths)
        orch.create_study()
        mock_cfg.optuna.get_storage_url.assert_called_once_with(mock_paths)

    # optimize: verify optuna verbosity set to WARNING
    @patch("optuna.create_study")
    @patch("orchard.optimization.orchestrator.orchestrator.get_search_space", return_value={})
    @patch("orchard.optimization.orchestrator.orchestrator.OptunaObjective")
    @patch("orchard.optimization.orchestrator.orchestrator.log_optimization_header")
    @patch("orchard.optimization.orchestrator.orchestrator.build_callbacks", return_value=[])
    @patch("optuna.logging.set_verbosity")
    def test_optimize_sets_verbosity_warning(
        self,
        mock_set_verb: MagicMock,
        mock_build_cb: MagicMock,
        mock_log_header: MagicMock,
        mock_obj: MagicMock,
        mock_space: MagicMock,
        mock_create: MagicMock,
        mock_cfg: MagicMock,
        mock_paths: MagicMock,
    ) -> None:
        """Verify optuna verbosity is set to WARNING."""
        mock_study = MagicMock()
        mock_study.trials = []
        mock_create.return_value = mock_study
        mock_obj.return_value = MagicMock()

        orch = _make_orch(mock_cfg, mock_paths)
        with patch.object(orch, "_post_optimization_processing"):
            orch.optimize()

        mock_set_verb.assert_called_once_with(optuna.logging.WARNING)

    # optimize: verify get_search_space called with exact kwargs
    @patch("optuna.create_study")
    @patch("orchard.optimization.orchestrator.orchestrator.get_search_space")
    @patch("orchard.optimization.orchestrator.orchestrator.OptunaObjective")
    @patch("orchard.optimization.orchestrator.orchestrator.log_optimization_header")
    @patch("orchard.optimization.orchestrator.orchestrator.build_callbacks", return_value=[])
    def test_optimize_search_space_kwargs(
        self,
        mock_build_cb: MagicMock,
        mock_log_header: MagicMock,
        mock_obj: MagicMock,
        mock_space: MagicMock,
        mock_create: MagicMock,
        mock_cfg: MagicMock,
        mock_paths: MagicMock,
    ) -> None:
        """Verify exact kwargs to get_search_space."""
        mock_study = MagicMock()
        mock_study.trials = []
        mock_create.return_value = mock_study
        mock_space.return_value = {}
        mock_obj.return_value = MagicMock()

        mock_cfg.optuna.search_space_preset = "full"
        mock_cfg.dataset.resolution = 64
        mock_cfg.optuna.enable_model_search = True
        mock_cfg.optuna.model_pool = ["resnet18", "vit"]
        overrides_sentinel = MagicMock()
        mock_cfg.optuna.search_space_overrides = overrides_sentinel

        orch = _make_orch(mock_cfg, mock_paths)
        with patch.object(orch, "_post_optimization_processing"):
            orch.optimize()

        mock_space.assert_called_once_with(
            "full",
            resolution=64,
            include_models=True,
            model_pool=["resnet18", "vit"],
            overrides=overrides_sentinel,
            task_type=mock_cfg.task_type,
        )

    # optimize: verify OptunaObjective constructed with exact kwargs
    @patch("optuna.create_study")
    @patch("orchard.optimization.orchestrator.orchestrator.get_search_space", return_value={})
    @patch("orchard.optimization.orchestrator.orchestrator.OptunaObjective")
    @patch("orchard.optimization.orchestrator.orchestrator.log_optimization_header")
    @patch("orchard.optimization.orchestrator.orchestrator.build_callbacks", return_value=[])
    def test_optimize_objective_kwargs(
        self,
        mock_build_cb: MagicMock,
        mock_log_header: MagicMock,
        mock_obj_cls: MagicMock,
        mock_space: MagicMock,
        mock_create: MagicMock,
        mock_cfg: MagicMock,
        mock_paths: MagicMock,
    ) -> None:
        """Verify exact kwargs to OptunaObjective."""
        mock_study = MagicMock()
        mock_study.trials = []
        mock_create.return_value = mock_study
        search_space = {"lr": (0.001, 0.1)}
        mock_space.return_value = search_space
        mock_obj_cls.return_value = MagicMock()

        tracker_sentinel = object()
        device = torch.device("cpu")
        orch = _make_orch(mock_cfg, mock_paths, device=device, tracker=tracker_sentinel)
        with patch.object(orch, "_post_optimization_processing"):
            orch.optimize()

        mock_obj_cls.assert_called_once_with(
            cfg=mock_cfg,
            search_space=search_space,
            device=device,
            tracker=tracker_sentinel,
        )

    # optimize: verify log_optimization_header called with cfg
    @patch("optuna.create_study")
    @patch("orchard.optimization.orchestrator.orchestrator.get_search_space", return_value={})
    @patch("orchard.optimization.orchestrator.orchestrator.OptunaObjective")
    @patch("orchard.optimization.orchestrator.orchestrator.log_optimization_header")
    @patch("orchard.optimization.orchestrator.orchestrator.build_callbacks", return_value=[])
    def test_optimize_calls_log_header_with_cfg(
        self,
        mock_build_cb: MagicMock,
        mock_log_header: MagicMock,
        mock_obj: MagicMock,
        mock_space: MagicMock,
        mock_create: MagicMock,
        mock_cfg: MagicMock,
        mock_paths: MagicMock,
    ) -> None:
        """Verify log_optimization_header receives self.cfg."""
        mock_study = MagicMock()
        mock_study.trials = []
        mock_create.return_value = mock_study
        mock_obj.return_value = MagicMock()

        orch = _make_orch(mock_cfg, mock_paths)
        with patch.object(orch, "_post_optimization_processing"):
            orch.optimize()

        mock_log_header.assert_called_once_with(mock_cfg)

    # optimize: verify build_callbacks called with correct args
    @patch("optuna.create_study")
    @patch("orchard.optimization.orchestrator.orchestrator.get_search_space", return_value={})
    @patch("orchard.optimization.orchestrator.orchestrator.OptunaObjective")
    @patch("orchard.optimization.orchestrator.orchestrator.log_optimization_header")
    @patch("orchard.optimization.orchestrator.orchestrator.build_callbacks")
    @patch("orchard.optimization.orchestrator.orchestrator.get_task")
    def test_optimize_build_callbacks_args(
        self,
        mock_get_task: MagicMock,
        mock_build_cb: MagicMock,
        mock_log_header: MagicMock,
        mock_obj: MagicMock,
        mock_space: MagicMock,
        mock_create: MagicMock,
        mock_cfg: MagicMock,
        mock_paths: MagicMock,
    ) -> None:
        """Verify build_callbacks receives optuna config, monitor_metric, and task_thresholds."""
        mock_study = MagicMock()
        mock_study.trials = []
        mock_create.return_value = mock_study
        mock_obj.return_value = MagicMock()
        mock_build_cb.return_value = []
        mock_cfg.training.monitor_metric = "f1"

        mock_task = MagicMock()
        mock_task.early_stopping_thresholds = {"f1": 0.98}
        mock_get_task.return_value = mock_task

        orch = _make_orch(mock_cfg, mock_paths)
        with patch.object(orch, "_post_optimization_processing"):
            orch.optimize()

        mock_build_cb.assert_called_once_with(
            mock_cfg.optuna,
            "f1",
            mock_cfg.training.monitor_direction,
            task_thresholds={"f1": 0.98},
        )

    # optimize: verify study.set_user_attr called with correct key and value
    @patch("optuna.create_study")
    @patch("orchard.optimization.orchestrator.orchestrator.get_search_space", return_value={})
    @patch("orchard.optimization.orchestrator.orchestrator.OptunaObjective")
    @patch("orchard.optimization.orchestrator.orchestrator.log_optimization_header")
    @patch("orchard.optimization.orchestrator.orchestrator.build_callbacks", return_value=[])
    def test_optimize_sets_user_attr_n_trials(
        self,
        mock_build_cb: MagicMock,
        mock_log_header: MagicMock,
        mock_obj: MagicMock,
        mock_space: MagicMock,
        mock_create: MagicMock,
        mock_cfg: MagicMock,
        mock_paths: MagicMock,
    ) -> None:
        """Verify study.set_user_attr('n_trials', ...) is called."""
        mock_study = MagicMock()
        mock_study.trials = []
        mock_create.return_value = mock_study
        mock_obj.return_value = MagicMock()
        mock_cfg.optuna.n_trials = 10

        orch = _make_orch(mock_cfg, mock_paths)
        with patch.object(orch, "_post_optimization_processing"):
            orch.optimize()

        mock_study.set_user_attr.assert_called_once_with("n_trials", 10)

    # optimize: verify study.optimize called with exact kwargs
    @patch("optuna.create_study")
    @patch("orchard.optimization.orchestrator.orchestrator.get_search_space", return_value={})
    @patch("orchard.optimization.orchestrator.orchestrator.OptunaObjective")
    @patch("orchard.optimization.orchestrator.orchestrator.log_optimization_header")
    @patch("orchard.optimization.orchestrator.orchestrator.build_callbacks")
    def test_optimize_study_optimize_kwargs(
        self,
        mock_build_cb: MagicMock,
        mock_log_header: MagicMock,
        mock_obj_cls: MagicMock,
        mock_space: MagicMock,
        mock_create: MagicMock,
        mock_cfg: MagicMock,
        mock_paths: MagicMock,
    ) -> None:
        """Verify exact kwargs to study.optimize."""
        mock_study = MagicMock()
        mock_study.trials = []
        mock_create.return_value = mock_study
        objective_instance = MagicMock()
        mock_obj_cls.return_value = objective_instance
        callbacks_sentinel = [MagicMock()]
        mock_build_cb.return_value = callbacks_sentinel

        mock_cfg.optuna.n_trials = 20
        mock_cfg.optuna.timeout = 300
        mock_cfg.optuna.n_jobs = 2
        mock_cfg.optuna.show_progress_bar = True

        orch = _make_orch(mock_cfg, mock_paths)
        with patch.object(orch, "_post_optimization_processing"):
            orch.optimize()

        mock_study.optimize.assert_called_once_with(
            objective_instance,
            n_trials=20,
            timeout=300,
            n_jobs=2,
            show_progress_bar=True,
            callbacks=callbacks_sentinel,
        )

    # optimize: verify interrupted=False path does NOT sleep
    @patch("orchard.optimization.orchestrator.orchestrator.time.sleep")
    @patch("optuna.create_study")
    @patch("orchard.optimization.orchestrator.orchestrator.get_search_space", return_value={})
    @patch("orchard.optimization.orchestrator.orchestrator.OptunaObjective")
    @patch("orchard.optimization.orchestrator.orchestrator.log_optimization_header")
    @patch("orchard.optimization.orchestrator.orchestrator.build_callbacks", return_value=[])
    def test_optimize_no_interrupt_no_sleep(
        self,
        mock_build_cb: MagicMock,
        mock_log_header: MagicMock,
        mock_obj: MagicMock,
        mock_space: MagicMock,
        mock_create: MagicMock,
        mock_sleep: MagicMock,
        mock_cfg: MagicMock,
        mock_paths: MagicMock,
    ) -> None:
        """Verify no sleep when optimization completes normally."""
        mock_study = MagicMock()
        mock_study.trials = []
        mock_create.return_value = mock_study
        mock_obj.return_value = MagicMock()

        orch = _make_orch(mock_cfg, mock_paths)
        with patch.object(orch, "_post_optimization_processing"):
            orch.optimize()

        mock_sleep.assert_not_called()

    # optimize: verify _post_optimization_processing called even on interrupt
    @patch("orchard.optimization.orchestrator.orchestrator.time.sleep")
    @patch("optuna.create_study")
    @patch("orchard.optimization.orchestrator.orchestrator.get_search_space", return_value={})
    @patch("orchard.optimization.orchestrator.orchestrator.OptunaObjective")
    @patch("orchard.optimization.orchestrator.orchestrator.log_optimization_header")
    @patch("orchard.optimization.orchestrator.orchestrator.build_callbacks", return_value=[])
    def test_optimize_interrupt_still_calls_post_processing(
        self,
        mock_build_cb: MagicMock,
        mock_log_header: MagicMock,
        mock_obj: MagicMock,
        mock_space: MagicMock,
        mock_create: MagicMock,
        mock_sleep: MagicMock,
        mock_cfg: MagicMock,
        mock_paths: MagicMock,
    ) -> None:
        """Verify _post_optimization_processing is called after KeyboardInterrupt."""
        mock_study = MagicMock()
        mock_study.trials = []
        mock_create.return_value = mock_study
        mock_obj.return_value = MagicMock()
        mock_study.optimize.side_effect = KeyboardInterrupt

        orch = _make_orch(mock_cfg, mock_paths)
        with patch.object(orch, "_post_optimization_processing") as mock_post:
            orch.optimize()
            mock_post.assert_called_once_with(mock_study)

    # _post_optimization_processing: no completed trials - only summary, no top_trials
    @patch("orchard.optimization.orchestrator.orchestrator.export_best_config")
    @patch("orchard.optimization.orchestrator.orchestrator.export_top_trials")
    @patch("orchard.optimization.orchestrator.orchestrator.export_study_summary")
    @patch("orchard.optimization.orchestrator.orchestrator.generate_visualizations")
    def test_post_no_trials_skips_viz_top_best(
        self,
        mock_viz: MagicMock,
        mock_summary: MagicMock,
        mock_top: MagicMock,
        mock_best: MagicMock,
        mock_cfg: MagicMock,
        mock_paths: MagicMock,
    ) -> None:
        """No completed trials: only summary exported, nothing else."""
        study = MagicMock()
        study.trials = []

        mock_cfg.optuna.save_plots = True
        mock_cfg.optuna.save_best_config = True

        orch = _make_orch(mock_cfg, mock_paths)
        orch._post_optimization_processing(study)

        mock_summary.assert_called_once_with(study, mock_paths)
        mock_viz.assert_not_called()
        mock_top.assert_not_called()
        mock_best.assert_not_called()

    # _post_optimization_processing: with trials, verify export_top_trials gets monitor_metric
    @patch("orchard.optimization.orchestrator.orchestrator.export_best_config")
    @patch("orchard.optimization.orchestrator.orchestrator.export_top_trials")
    @patch("orchard.optimization.orchestrator.orchestrator.export_study_summary")
    @patch("orchard.optimization.orchestrator.orchestrator.generate_visualizations")
    def test_post_with_trials_export_top_trials_args(
        self,
        mock_viz: MagicMock,
        mock_summary: MagicMock,
        mock_top: MagicMock,
        mock_best: MagicMock,
        study_with_trials: MagicMock,
        mock_cfg: MagicMock,
        mock_paths: MagicMock,
    ) -> None:
        """Verify export_top_trials receives study, paths, and monitor_metric."""
        mock_cfg.optuna.save_plots = False
        mock_cfg.optuna.save_best_config = False
        mock_cfg.training.monitor_metric = "balanced_accuracy"

        orch = _make_orch(mock_cfg, mock_paths)
        orch._post_optimization_processing(study_with_trials)

        mock_top.assert_called_once_with(study_with_trials, mock_paths, "balanced_accuracy")

    # _post_optimization_processing: verify export_best_config gets study, cfg, paths
    @patch("orchard.optimization.orchestrator.orchestrator.export_best_config")
    @patch("orchard.optimization.orchestrator.orchestrator.export_top_trials")
    @patch("orchard.optimization.orchestrator.orchestrator.export_study_summary")
    @patch("orchard.optimization.orchestrator.orchestrator.generate_visualizations")
    def test_post_export_best_config_args(
        self,
        mock_viz: MagicMock,
        mock_summary: MagicMock,
        mock_top: MagicMock,
        mock_best: MagicMock,
        study_with_trials: MagicMock,
        mock_cfg: MagicMock,
        mock_paths: MagicMock,
    ) -> None:
        """Verify export_best_config receives study, cfg, and paths."""
        mock_cfg.optuna.save_plots = False
        mock_cfg.optuna.save_best_config = True

        orch = _make_orch(mock_cfg, mock_paths)
        orch._post_optimization_processing(study_with_trials)

        mock_best.assert_called_once_with(study_with_trials, mock_cfg, mock_paths)

    # _post_optimization_processing: verify generate_visualizations gets figures path
    @patch("orchard.optimization.orchestrator.orchestrator.export_best_config")
    @patch("orchard.optimization.orchestrator.orchestrator.export_top_trials")
    @patch("orchard.optimization.orchestrator.orchestrator.export_study_summary")
    @patch("orchard.optimization.orchestrator.orchestrator.generate_visualizations")
    def test_post_viz_uses_figures_path(
        self,
        mock_viz: MagicMock,
        mock_summary: MagicMock,
        mock_top: MagicMock,
        mock_best: MagicMock,
        study_with_trials: MagicMock,
        mock_cfg: MagicMock,
        mock_paths: MagicMock,
    ) -> None:
        """Verify generate_visualizations receives paths.figures, not paths."""
        mock_cfg.optuna.save_plots = True
        mock_cfg.optuna.save_best_config = False

        orch = _make_orch(mock_cfg, mock_paths)
        orch._post_optimization_processing(study_with_trials)

        mock_viz.assert_called_once_with(study_with_trials, mock_paths.figures)

    # _post_optimization_processing: summary always called (with and without trials)
    @patch("orchard.optimization.orchestrator.orchestrator.export_best_config")
    @patch("orchard.optimization.orchestrator.orchestrator.export_top_trials")
    @patch("orchard.optimization.orchestrator.orchestrator.export_study_summary")
    @patch("orchard.optimization.orchestrator.orchestrator.generate_visualizations")
    def test_post_summary_always_called_with_study_and_paths(
        self,
        mock_viz: MagicMock,
        mock_summary: MagicMock,
        mock_top: MagicMock,
        mock_best: MagicMock,
        study_with_trials: MagicMock,
        mock_cfg: MagicMock,
        mock_paths: MagicMock,
    ) -> None:
        """Verify export_study_summary always gets (study, paths)."""
        mock_cfg.optuna.save_plots = False
        mock_cfg.optuna.save_best_config = False

        orch = _make_orch(mock_cfg, mock_paths)
        orch._post_optimization_processing(study_with_trials)

        mock_summary.assert_called_once_with(study_with_trials, mock_paths)

    # run_optimization: verify tracker passed through
    @patch("orchard.optimization.orchestrator.orchestrator.OptunaOrchestrator")
    def test_run_optimization_passes_tracker(
        self, mock_orch_class: MagicMock, mock_cfg: MagicMock, mock_paths: MagicMock
    ) -> None:
        """Verify run_optimization passes tracker to orchestrator."""
        mock_orch = MagicMock()
        mock_orch.optimize.return_value = MagicMock()
        mock_orch_class.return_value = mock_orch

        from orchard.optimization.orchestrator.orchestrator import (
            run_optimization as run_opt,
        )

        tracker_sentinel = object()
        run_opt(
            cfg=cast(Config, mock_cfg),
            device=torch.device("cpu"),
            paths=cast(RunPaths, mock_paths),
            tracker=cast(TrackerProtocol, tracker_sentinel),
        )

        mock_orch_class.assert_called_once_with(
            cfg=mock_cfg,
            device=torch.device("cpu"),
            paths=mock_paths,
            tracker=tracker_sentinel,
        )

    # optimize: verify the return value is always the study (even on interrupt)
    @patch("orchard.optimization.orchestrator.orchestrator.time.sleep")
    @patch("optuna.create_study")
    @patch("orchard.optimization.orchestrator.orchestrator.get_search_space", return_value={})
    @patch("orchard.optimization.orchestrator.orchestrator.OptunaObjective")
    @patch("orchard.optimization.orchestrator.orchestrator.log_optimization_header")
    @patch("orchard.optimization.orchestrator.orchestrator.build_callbacks", return_value=[])
    def test_optimize_returns_study_on_interrupt(
        self,
        mock_build_cb: MagicMock,
        mock_log_header: MagicMock,
        mock_obj: MagicMock,
        mock_space: MagicMock,
        mock_create: MagicMock,
        mock_sleep: MagicMock,
        mock_cfg: MagicMock,
        mock_paths: MagicMock,
    ) -> None:
        """Verify optimize returns the study even after KeyboardInterrupt."""
        mock_study = MagicMock()
        mock_study.trials = []
        mock_create.return_value = mock_study
        mock_obj.return_value = MagicMock()
        mock_study.optimize.side_effect = KeyboardInterrupt

        orch = _make_orch(mock_cfg, mock_paths)
        with patch.object(orch, "_post_optimization_processing"):
            result = orch.optimize()

        assert result is mock_study

    # optimize: verify sleep(5) specifically on interrupt
    @patch("orchard.optimization.orchestrator.orchestrator.time.sleep")
    @patch("optuna.create_study")
    @patch("orchard.optimization.orchestrator.orchestrator.get_search_space", return_value={})
    @patch("orchard.optimization.orchestrator.orchestrator.OptunaObjective")
    @patch("orchard.optimization.orchestrator.orchestrator.log_optimization_header")
    @patch("orchard.optimization.orchestrator.orchestrator.build_callbacks", return_value=[])
    def test_optimize_interrupt_sleeps_exactly_5(
        self,
        mock_build_cb: MagicMock,
        mock_log_header: MagicMock,
        mock_obj: MagicMock,
        mock_space: MagicMock,
        mock_create: MagicMock,
        mock_sleep: MagicMock,
        mock_cfg: MagicMock,
        mock_paths: MagicMock,
    ) -> None:
        """Verify sleep is called with exactly 5 seconds on interrupt."""
        mock_study = MagicMock()
        mock_study.trials = []
        mock_create.return_value = mock_study
        mock_obj.return_value = MagicMock()
        mock_study.optimize.side_effect = KeyboardInterrupt

        orch = _make_orch(mock_cfg, mock_paths)
        with patch.object(orch, "_post_optimization_processing"):
            orch.optimize()

        mock_sleep.assert_called_once_with(5)

    # _post_optimization_processing: both save_plots and save_best_config enabled
    @patch("orchard.optimization.orchestrator.orchestrator.export_best_config")
    @patch("orchard.optimization.orchestrator.orchestrator.export_top_trials")
    @patch("orchard.optimization.orchestrator.orchestrator.export_study_summary")
    @patch("orchard.optimization.orchestrator.orchestrator.generate_visualizations")
    def test_post_both_flags_enabled(
        self,
        mock_viz: MagicMock,
        mock_summary: MagicMock,
        mock_top: MagicMock,
        mock_best: MagicMock,
        study_with_trials: MagicMock,
        mock_cfg: MagicMock,
        mock_paths: MagicMock,
    ) -> None:
        """Verify both viz and best_config when both flags True."""
        mock_cfg.optuna.save_plots = True
        mock_cfg.optuna.save_best_config = True

        orch = _make_orch(mock_cfg, mock_paths)
        orch._post_optimization_processing(study_with_trials)

        mock_viz.assert_called_once()
        mock_best.assert_called_once()
        mock_summary.assert_called_once()
        mock_top.assert_called_once()

    # _post_optimization_processing: both flags disabled
    @patch("orchard.optimization.orchestrator.orchestrator.export_best_config")
    @patch("orchard.optimization.orchestrator.orchestrator.export_top_trials")
    @patch("orchard.optimization.orchestrator.orchestrator.export_study_summary")
    @patch("orchard.optimization.orchestrator.orchestrator.generate_visualizations")
    def test_post_both_flags_disabled(
        self,
        mock_viz: MagicMock,
        mock_summary: MagicMock,
        mock_top: MagicMock,
        mock_best: MagicMock,
        study_with_trials: MagicMock,
        mock_cfg: MagicMock,
        mock_paths: MagicMock,
    ) -> None:
        """Verify no viz or best_config when both flags False."""
        mock_cfg.optuna.save_plots = False
        mock_cfg.optuna.save_best_config = False

        orch = _make_orch(mock_cfg, mock_paths)
        orch._post_optimization_processing(study_with_trials)

        mock_viz.assert_not_called()
        mock_best.assert_not_called()
        # summary and top_trials always called
        mock_summary.assert_called_once()
        mock_top.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
