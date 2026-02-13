"""
Test Suite for Optuna Orchestrator Module.

Focused on testing the orchestrator logic with proper mocking
to avoid triggering real downloads, file I/O, or network calls.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import optuna
import pytest

from orchard.optimization.orchestrator import OptunaOrchestrator
from orchard.optimization.orchestrator.builders import (
    build_callbacks,
    build_pruner,
    build_sampler,
)
from orchard.optimization.orchestrator.config import (
    PRUNER_REGISTRY,
    SAMPLER_REGISTRY,
    TRAINING_PARAMS,
    map_param_to_config_path,
)
from orchard.optimization.orchestrator.exporters import (
    build_best_config_dict,
    build_trial_data,
)
from orchard.optimization.orchestrator.utils import (
    get_completed_trials,
    has_completed_trials,
)


# FIXTURES
@pytest.fixture
def mock_cfg():
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
    cfg.optuna.metric_name = "auc"
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
def mock_paths(tmp_path):
    """Mock RunPaths with temp dirs."""
    paths = MagicMock()
    paths.root = tmp_path
    paths.reports = tmp_path / "reports"
    paths.figures = tmp_path / "figures"
    paths.reports.mkdir(exist_ok=True)
    paths.figures.mkdir(exist_ok=True)
    return paths


@pytest.fixture
def completed_trial():
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
def study_with_trials(completed_trial):
    """Mock study with one completed trial."""
    study = MagicMock()
    study.study_name = "test"
    study.direction = optuna.study.StudyDirection.MAXIMIZE
    study.trials = [completed_trial]
    study.best_trial = completed_trial
    study.best_params = completed_trial.params
    study.best_value = completed_trial.value
    return study


# UNIT TESTS: config.py
@pytest.mark.unit
class TestConfig:
    """Test config constants and functions."""

    def test_sampler_registry_has_tpe(self):
        """Verify TPE sampler is registered."""
        assert "tpe" in SAMPLER_REGISTRY
        assert SAMPLER_REGISTRY["tpe"] == optuna.samplers.TPESampler

    def test_pruner_registry_has_median(self):
        """Verify Median pruner is registered."""
        assert "median" in PRUNER_REGISTRY

    def test_training_params_includes_lr(self):
        """Verify learning_rate is a training param."""
        assert "learning_rate" in TRAINING_PARAMS

    def test_map_param_to_training(self):
        """Test mapping learning_rate to training section."""
        section, key = map_param_to_config_path("learning_rate")
        assert section == "training"
        assert key == "learning_rate"


# UNIT TESTS: builders.py
@pytest.mark.unit
class TestBuilders:
    """Test builder functions."""

    def test_build_sampler_tpe(self, mock_cfg):
        """Test building TPE sampler."""
        sampler = build_sampler("tpe", mock_cfg)
        assert isinstance(sampler, optuna.samplers.TPESampler)

    def test_build_pruner_median(self, mock_cfg):
        """Test building Median pruner."""
        pruner = build_pruner(True, "median", mock_cfg)
        assert isinstance(pruner, optuna.pruners.MedianPruner)

    def test_build_pruner_disabled(self, mock_cfg):
        """Test disabled pruning returns NopPruner."""
        pruner = build_pruner(False, "median", mock_cfg)
        assert isinstance(pruner, optuna.pruners.BasePruner)

    @patch("orchard.optimization.orchestrator.builders.get_early_stopping_callback")
    def test_build_callbacks(self, mock_callback_fn, mock_cfg):
        """Test building callbacks list."""
        mock_callback_fn.return_value = None
        callbacks = build_callbacks(mock_cfg)
        assert isinstance(callbacks, list)


# UNIT TESTS: utils.py
@pytest.mark.unit
class TestUtils:
    """Test utility functions."""

    def test_get_completed_trials(self, study_with_trials):
        """Test extracting completed trials."""
        completed = get_completed_trials(study_with_trials)
        assert len(completed) == 1

    def test_has_completed_trials_true(self, study_with_trials):
        """Test has_completed_trials returns True."""
        assert has_completed_trials(study_with_trials) is True

    def test_has_completed_trials_false(self):
        """Test has_completed_trials returns False."""
        study = MagicMock()
        study.trials = []
        assert has_completed_trials(study) is False


# TESTS: exporters.py
@pytest.mark.unit
class TestExporters:
    """Test exporter functions."""

    def test_build_trial_data(self, completed_trial):
        """Test building trial data dict."""
        data = build_trial_data(completed_trial)
        assert data["number"] == 1
        assert data["value"] == pytest.approx(0.95)
        assert data["state"] == "COMPLETE"

    def test_build_best_config_dict(self, mock_cfg):
        """Test building config dict from params."""
        params = {"learning_rate": 0.001}
        config_dict = build_best_config_dict(params, mock_cfg)
        assert config_dict["training"]["learning_rate"] == pytest.approx(0.001)


# INTEGRATION TESTS: OptunaOrchestrator
@pytest.mark.integration
class TestOptunaOrchestrator:
    """Integration tests for orchestrator."""

    def test_init(self, mock_cfg, mock_paths):
        """Test orchestrator initialization."""
        orch = OptunaOrchestrator(cfg=mock_cfg, device="cpu", paths=mock_paths)
        assert orch.cfg == mock_cfg
        assert orch.device == "cpu"
        assert orch.paths == mock_paths

    @patch("optuna.create_study")
    def test_create_study(self, mock_create, mock_cfg, mock_paths):
        """Test create_study builds components."""
        mock_study = MagicMock()
        mock_create.return_value = mock_study

        orch = OptunaOrchestrator(cfg=mock_cfg, device="cpu", paths=mock_paths)
        study = orch.create_study()

        mock_create.assert_called_once()
        assert study == mock_study

    @patch("orchard.optimization.orchestrator.orchestrator.export_study_summary")
    def test_post_optimization_no_trials(self, mock_export, mock_cfg, mock_paths):
        """Test post-optimization with no completed trials."""
        study = MagicMock()
        study.trials = []
        study.study_name = "test"
        study.direction = MagicMock()
        study.direction.name = "MAXIMIZE"

        orch = OptunaOrchestrator(cfg=mock_cfg, device="cpu", paths=mock_paths)
        orch._post_optimization_processing(study)

        mock_export.assert_called_once()

    @patch("orchard.optimization.orchestrator.orchestrator.export_study_summary")
    @patch("orchard.optimization.orchestrator.orchestrator.export_top_trials")
    def test_post_optimization_with_trials(
        self, mock_top_trials, mock_summary, study_with_trials, mock_cfg, mock_paths
    ):
        """Test post-optimization with completed trials."""
        orch = OptunaOrchestrator(cfg=mock_cfg, device="cpu", paths=mock_paths)
        orch._post_optimization_processing(study_with_trials)

        mock_summary.assert_called_once()
        mock_top_trials.assert_called_once()

    @patch("optuna.create_study")
    @patch("orchard.optimization.orchestrator.orchestrator.get_search_space")
    @patch("orchard.optimization.orchestrator.orchestrator.OptunaObjective")
    def test_optimize_full_flow(self, mock_obj, mock_space, mock_create, mock_cfg, mock_paths):
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

        orch = OptunaOrchestrator(cfg=mock_cfg, device="cpu", paths=mock_paths)

        with patch.object(orch, "_post_optimization_processing"):
            result = orch.optimize()

        mock_create.assert_called_once()
        mock_space.assert_called_once()
        mock_obj.assert_called_once()
        mock_study.optimize.assert_called_once()
        assert result == mock_study

    @patch("orchard.optimization.orchestrator.orchestrator.OptunaOrchestrator")
    def test_run_optimization(self, mock_orch_class, mock_cfg, mock_paths):
        """Test run_optimization convenience function."""
        mock_orch = MagicMock()
        mock_study = MagicMock()
        mock_orch.optimize.return_value = mock_study
        mock_orch_class.return_value = mock_orch

        from orchard.optimization.orchestrator.orchestrator import run_optimization as run_opt

        result = run_opt(cfg=mock_cfg, device="cpu", paths=mock_paths)

        mock_orch_class.assert_called_once_with(cfg=mock_cfg, device="cpu", paths=mock_paths)
        mock_orch.optimize.assert_called_once()
        assert result == mock_study

    @patch("orchard.optimization.orchestrator.orchestrator.generate_visualizations")
    @patch("orchard.optimization.orchestrator.orchestrator.export_best_config")
    @patch("orchard.optimization.orchestrator.orchestrator.export_study_summary")
    def test_post_optimization_with_save_plots_enabled(
        self,
        mock_export_summary,
        mock_export_best,
        mock_generate_viz,
        study_with_trials,
        mock_cfg,
        mock_paths,
    ):
        """Test post-optimization with save_plots enabled."""
        mock_cfg.optuna.save_plots = True
        mock_cfg.optuna.save_best_config = False

        orch = OptunaOrchestrator(cfg=mock_cfg, device="cpu", paths=mock_paths)
        orch._post_optimization_processing(study_with_trials)

        mock_generate_viz.assert_called_once_with(study_with_trials, mock_paths.figures)
        mock_export_summary.assert_called_once()
        mock_export_best.assert_not_called()

    @patch("orchard.optimization.orchestrator.orchestrator.generate_visualizations")
    @patch("orchard.optimization.orchestrator.orchestrator.export_best_config")
    @patch("orchard.optimization.orchestrator.orchestrator.export_study_summary")
    def test_post_optimization_with_save_best_config_enabled(
        self,
        mock_export_summary,
        mock_export_best,
        mock_generate_viz,
        study_with_trials,
        mock_cfg,
        mock_paths,
    ):
        """Test post-optimization with save_best_config enabled."""
        mock_cfg.optuna.save_plots = False
        mock_cfg.optuna.save_best_config = True

        orch = OptunaOrchestrator(cfg=mock_cfg, device="cpu", paths=mock_paths)
        orch._post_optimization_processing(study_with_trials)

        mock_export_best.assert_called_once_with(study_with_trials, mock_cfg, mock_paths)
        mock_export_summary.assert_called_once()
        mock_generate_viz.assert_not_called()

    @patch("orchard.optimization.orchestrator.orchestrator.OptunaObjective")
    @patch("orchard.optimization.orchestrator.orchestrator.get_search_space")
    @patch("optuna.create_study")
    def test_optimize_keyboard_interrupt(
        self, mock_create_study, mock_get_search_space, mock_objective_class, mock_cfg, mock_paths
    ):
        """Test optimize handles KeyboardInterrupt."""
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

        orch = OptunaOrchestrator(cfg=mock_cfg, device="cpu", paths=mock_paths)

        with patch.object(orch, "_post_optimization_processing") as mock_post:
            result = orch.optimize()

            mock_post.assert_called_once_with(mock_study)
            assert result == mock_study


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
