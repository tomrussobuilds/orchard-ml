"""
Smoke Tests for Optuna Orchestrator Module.

Minimal tests to validate orchestrator initialization and core methods.
These are essential smoke tests to boost coverage from 0% to ~15%.
"""

# =========================================================================== #
#                         Standard Imports                                    #
# =========================================================================== #
from unittest.mock import MagicMock, patch

# =========================================================================== #
#                         Third-Party Imports                                 #
# =========================================================================== #
import optuna
import pytest

# =========================================================================== #
#                         Internal Imports                                    #
# =========================================================================== #
from orchard.optimization.orchestrator import OptunaOrchestrator, run_optimization

# =========================================================================== #
#                    ORCHESTRATOR: INITIALIZATION                             #
# =========================================================================== #


@pytest.mark.unit
def test_orchestrator_init():
    """Test OptunaOrchestrator initializes correctly."""
    mock_cfg = MagicMock()
    mock_device = MagicMock()
    mock_paths = MagicMock()
    mock_paths.root = "/tmp/outputs"

    orchestrator = OptunaOrchestrator(cfg=mock_cfg, device=mock_device, paths=mock_paths)

    assert orchestrator.cfg == mock_cfg
    assert orchestrator.device == mock_device
    assert orchestrator.paths == mock_paths


# =========================================================================== #
#                    ORCHESTRATOR: SAMPLER CONFIGURATION                      #
# =========================================================================== #


@pytest.mark.unit
@pytest.mark.parametrize(
    "sampler_type,expected_class",
    [
        ("tpe", optuna.samplers.TPESampler),
        ("cmaes", optuna.samplers.CmaEsSampler),
        ("random", optuna.samplers.RandomSampler),
    ],
)
def test_get_sampler_valid_types(sampler_type, expected_class):
    """Test _get_sampler creates correct sampler types."""
    mock_cfg = MagicMock()
    mock_cfg.optuna.sampler_type = sampler_type
    mock_paths = MagicMock()

    orchestrator = OptunaOrchestrator(cfg=mock_cfg, device=MagicMock(), paths=mock_paths)

    sampler = orchestrator._get_sampler()

    assert isinstance(sampler, expected_class)


@pytest.mark.unit
def test_get_sampler_invalid_type():
    """Test _get_sampler raises error for unknown sampler."""
    mock_cfg = MagicMock()
    mock_cfg.optuna.sampler_type = "invalid_sampler"
    mock_paths = MagicMock()

    orchestrator = OptunaOrchestrator(cfg=mock_cfg, device=MagicMock(), paths=mock_paths)

    with pytest.raises(ValueError, match="Unknown sampler"):
        orchestrator._get_sampler()


# =========================================================================== #
#                    ORCHESTRATOR: PRUNER CONFIGURATION                       #
# =========================================================================== #


@pytest.mark.unit
def test_get_pruner_disabled():
    """Test _get_pruner returns NopPruner when pruning disabled."""
    mock_cfg = MagicMock()
    mock_cfg.optuna.enable_pruning = False
    mock_paths = MagicMock()

    orchestrator = OptunaOrchestrator(cfg=mock_cfg, device=MagicMock(), paths=mock_paths)

    pruner = orchestrator._get_pruner()

    assert isinstance(pruner, optuna.pruners.NopPruner)


@pytest.mark.unit
@pytest.mark.parametrize(
    "pruner_type,expected_class",
    [
        ("median", optuna.pruners.MedianPruner),
        ("hyperband", optuna.pruners.HyperbandPruner),
        ("none", optuna.pruners.NopPruner),
    ],
)
def test_get_pruner_valid_types(pruner_type, expected_class):
    """Test _get_pruner creates correct pruner types."""
    mock_cfg = MagicMock()
    mock_cfg.optuna.enable_pruning = True
    mock_cfg.optuna.pruner_type = pruner_type
    mock_paths = MagicMock()

    orchestrator = OptunaOrchestrator(cfg=mock_cfg, device=MagicMock(), paths=mock_paths)

    pruner = orchestrator._get_pruner()

    assert isinstance(pruner, expected_class)


@pytest.mark.unit
def test_get_pruner_invalid_type():
    """Test _get_pruner raises error for unknown pruner."""
    mock_cfg = MagicMock()
    mock_cfg.optuna.enable_pruning = True
    mock_cfg.optuna.pruner_type = "invalid_pruner"
    mock_paths = MagicMock()

    orchestrator = OptunaOrchestrator(cfg=mock_cfg, device=MagicMock(), paths=mock_paths)

    with pytest.raises(ValueError, match="Unknown pruner"):
        orchestrator._get_pruner()


# =========================================================================== #
#                    ORCHESTRATOR: STUDY CREATION                             #
# =========================================================================== #


@pytest.mark.unit
@patch("orchard.optimization.orchestrator.optuna.create_study")
def test_create_study(mock_create_study):
    """Test create_study configures and creates Optuna study."""
    mock_study = MagicMock()
    mock_create_study.return_value = mock_study

    mock_cfg = MagicMock()
    mock_cfg.optuna.study_name = "test_study"
    mock_cfg.optuna.direction = "maximize"
    mock_cfg.optuna.sampler_type = "tpe"
    mock_cfg.optuna.enable_pruning = False
    mock_cfg.optuna.load_if_exists = True
    mock_cfg.optuna.get_storage_url = MagicMock(return_value="sqlite:///test.db")
    mock_cfg.optuna.n_trials = 10

    mock_paths = MagicMock()

    orchestrator = OptunaOrchestrator(cfg=mock_cfg, device=MagicMock(), paths=mock_paths)

    study = orchestrator.create_study()

    assert study == mock_study
    mock_create_study.assert_called_once()


# =========================================================================== #
#                    CONVENIENCE FUNCTION                                     #
# =========================================================================== #


@pytest.mark.unit
@patch("orchard.optimization.orchestrator.OptunaOrchestrator")
def test_run_optimization_creates_orchestrator(mock_orchestrator_class):
    """Test run_optimization convenience function creates orchestrator."""
    mock_orchestrator = MagicMock()
    mock_orchestrator_class.return_value = mock_orchestrator
    mock_study = MagicMock()
    mock_orchestrator.optimize.return_value = mock_study

    mock_cfg = MagicMock()
    mock_device = MagicMock()
    mock_paths = MagicMock()

    result = run_optimization(cfg=mock_cfg, device=mock_device, paths=mock_paths)

    mock_orchestrator_class.assert_called_once_with(
        cfg=mock_cfg, device=mock_device, paths=mock_paths
    )
    mock_orchestrator.optimize.assert_called_once()
    assert result == mock_study


# =========================================================================== #
#                    EXPORT METHODS: EDGE CASES                               #
# =========================================================================== #


@pytest.mark.unit
def test_export_best_config_no_completed_trials():
    """Test _export_best_config handles no completed trials gracefully."""
    mock_cfg = MagicMock()
    mock_paths = MagicMock()

    orchestrator = OptunaOrchestrator(cfg=mock_cfg, device=MagicMock(), paths=mock_paths)

    # Mock study with no completed trials
    mock_study = MagicMock()
    mock_trial = MagicMock()
    mock_trial.state = optuna.trial.TrialState.PRUNED
    mock_study.trials = [mock_trial]

    # Should handle gracefully without crashing
    with patch("orchard.optimization.orchestrator.logger") as mock_logger:
        orchestrator._export_best_config(mock_study)
        assert mock_logger.warning.called


@pytest.mark.unit
def test_generate_visualizations_no_completed_trials():
    """Test _generate_visualizations handles no completed trials gracefully."""
    mock_cfg = MagicMock()
    mock_paths = MagicMock()

    orchestrator = OptunaOrchestrator(cfg=mock_cfg, device=MagicMock(), paths=mock_paths)

    # Mock study with no completed trials
    mock_study = MagicMock()
    mock_trial = MagicMock()
    mock_trial.state = optuna.trial.TrialState.FAIL
    mock_study.trials = [mock_trial]

    # Should handle gracefully without crashing
    with patch("orchard.optimization.orchestrator.logger") as mock_logger:
        orchestrator._generate_visualizations(mock_study)
        assert mock_logger.warning.called


@pytest.mark.unit
def test_export_study_summary_no_completed_trials():
    """Test _export_study_summary handles no completed trials gracefully."""
    mock_cfg = MagicMock()
    mock_cfg.optuna.metric_name = "auc"
    mock_paths = MagicMock()
    mock_paths.reports = MagicMock()

    orchestrator = OptunaOrchestrator(cfg=mock_cfg, device=MagicMock(), paths=mock_paths)

    # Mock study with no completed trials
    mock_study = MagicMock()
    mock_study.study_name = "test"
    mock_study.direction.name = "MAXIMIZE"
    mock_trial = MagicMock()
    mock_trial.state = optuna.trial.TrialState.PRUNED
    mock_trial.number = 0
    mock_trial.value = None
    mock_trial.params = {}
    mock_trial.datetime_start = None
    mock_trial.datetime_complete = None
    mock_study.trials = [mock_trial]

    # Should save summary even with no completed trials
    with patch("builtins.open", MagicMock()):
        orchestrator._export_study_summary(mock_study)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
