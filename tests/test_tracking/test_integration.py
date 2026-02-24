"""
Test Suite for Tracker Integration Points.

Verifies that the tracker is correctly invoked in:
- ModelTrainer.train() (per-epoch logging)
- run_final_evaluation() (test metrics logging)
- OptunaObjective.__call__() (nested trial logging)
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from orchard.trainer import ModelTrainer

# --- FIXTURES ---


@pytest.fixture
def mock_cfg():
    """Mock Config with training parameters."""
    cfg = MagicMock()
    cfg.training.epochs = 3
    cfg.training.patience = 10
    cfg.training.use_amp = False
    cfg.training.mixup_alpha = 0.0
    cfg.training.mixup_epochs = 0
    cfg.training.grad_clip = 1.0
    cfg.training.monitor_metric = "auc"
    cfg.training.use_tqdm = False
    cfg.training.seed = 42
    return cfg


@pytest.fixture
def mock_tracker():
    """Mock tracker with all interface methods."""
    tracker = MagicMock()
    tracker.log_epoch = MagicMock()
    tracker.log_test_metrics = MagicMock()
    tracker.start_optuna_trial = MagicMock()
    tracker.end_optuna_trial = MagicMock()
    return tracker


# --- TRAINER INTEGRATION ---


@pytest.mark.unit
def test_trainer_calls_tracker_log_epoch(mock_cfg, mock_tracker):
    """ModelTrainer.train() calls tracker.log_epoch for each epoch."""
    model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))
    device = torch.device("cpu")

    batch = (torch.randn(4, 1, 28, 28), torch.randint(0, 10, (4,)))

    train_loader = MagicMock()
    train_loader.__iter__ = MagicMock(return_value=iter([batch]))
    train_loader.__len__ = MagicMock(return_value=1)

    val_loader = MagicMock()
    val_loader.__iter__ = MagicMock(return_value=iter([batch]))
    val_loader.__len__ = MagicMock(return_value=1)

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = ModelTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4),
            scheduler=torch.optim.lr_scheduler.StepLR(
                torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4),
                step_size=1,
            ),
            criterion=nn.CrossEntropyLoss(),
            device=device,
            cfg=mock_cfg,
            output_path=Path(tmpdir) / "best.pth",
            tracker=mock_tracker,
        )

        # Mock train/validate to avoid real computation
        with (
            patch("orchard.trainer._loop.train_one_epoch", return_value=0.5),
            patch(
                "orchard.trainer._loop.validate_epoch",
                return_value={"loss": 0.3, "accuracy": 0.9, "auc": 0.95},
            ),
        ):
            trainer.train()

    assert mock_tracker.log_epoch.call_count == mock_cfg.training.epochs

    # Verify call signature of first call
    first_call = mock_tracker.log_epoch.call_args_list[0]
    assert first_call[0][0] == 1  # epoch number
    assert isinstance(first_call[0][1], float)  # train_loss
    assert isinstance(first_call[0][2], dict)  # val_metrics
    assert isinstance(first_call[0][3], float)  # lr


# --- EVALUATION INTEGRATION ---


@pytest.mark.unit
def test_evaluation_calls_tracker_log_test_metrics(mock_tracker):
    """run_final_evaluation() calls tracker.log_test_metrics."""
    from orchard.evaluation.evaluation_pipeline import run_final_evaluation

    mock_model = MagicMock()
    mock_loader = MagicMock()
    mock_paths = MagicMock()
    mock_paths.best_model_path = Path("/mock/model.pth")
    mock_paths.final_report_path = Path("/mock/report.xlsx")
    mock_paths.logs = Path("/mock/logs")
    mock_paths.get_fig_path = MagicMock(return_value=Path("/mock/fig.png"))

    mock_cfg = MagicMock()
    mock_cfg.hardware.device = "cpu"
    mock_cfg.training.use_tta = False
    mock_cfg.dataset.metadata.is_anatomical = False
    mock_cfg.dataset.metadata.is_texture_based = False
    mock_cfg.architecture.name = "test"
    mock_cfg.dataset.resolution = 28

    test_metrics = {"accuracy": 0.92, "auc": 0.96}

    with (
        patch(
            "orchard.evaluation.evaluation_pipeline.evaluate_model",
            return_value=([], [], test_metrics, 0.88),
        ),
        patch("orchard.evaluation.evaluation_pipeline.plot_confusion_matrix"),
        patch("orchard.evaluation.evaluation_pipeline.plot_training_curves"),
        patch("orchard.evaluation.evaluation_pipeline.show_predictions"),
        patch("orchard.evaluation.evaluation_pipeline.create_structured_report") as mock_report,
    ):
        mock_report.return_value = MagicMock()

        macro_f1, test_acc, test_auc = run_final_evaluation(
            model=mock_model,
            test_loader=mock_loader,
            train_losses=[0.5, 0.4],
            val_metrics_history=[{"accuracy": 0.8}, {"accuracy": 0.9}],
            class_names=["a", "b"],
            paths=mock_paths,
            cfg=mock_cfg,
            tracker=mock_tracker,
        )

    mock_tracker.log_test_metrics.assert_called_once_with(test_acc=0.92, macro_f1=0.88)
    assert test_acc == pytest.approx(0.92, abs=1e-5)
    assert macro_f1 == pytest.approx(0.88, abs=1e-5)
    assert test_auc == pytest.approx(0.96, abs=1e-5)


# --- OPTUNA OBJECTIVE INTEGRATION ---


@pytest.mark.unit
def test_objective_calls_tracker_nested_runs(mock_tracker):
    """OptunaObjective.__call__() starts/ends nested MLflow runs."""
    from orchard.optimization.objective.objective import OptunaObjective

    mock_cfg = MagicMock()
    mock_cfg.optuna.metric_name = "auc"

    mock_trial = MagicMock()
    mock_trial.number = 0

    mock_search_space = MagicMock()
    mock_search_space.sample_params.return_value = {"lr": 0.01}

    # Mock all heavy dependencies
    mock_dataset = MagicMock()
    mock_dataloaders = MagicMock(return_value=(MagicMock(), MagicMock(), MagicMock()))
    mock_model_factory = MagicMock()

    objective = OptunaObjective(
        cfg=mock_cfg,
        search_space=mock_search_space,
        device=torch.device("cpu"),
        dataset_loader=MagicMock(return_value=mock_dataset),
        dataloader_factory=mock_dataloaders,
        model_factory=mock_model_factory,
        tracker=mock_tracker,
    )

    _mock_trial_cfg = MagicMock()
    _mock_trial_cfg.training.weighted_loss = False
    objective.config_builder.build = MagicMock(return_value=_mock_trial_cfg)

    # Mock executor to return a metric
    with patch(
        "orchard.optimization.objective.objective.TrialTrainingExecutor"
    ) as mock_executor_cls:
        mock_executor = MagicMock()
        mock_executor.execute.return_value = 0.95
        mock_executor_cls.return_value = mock_executor

        with patch("orchard.optimization.objective.objective.log_trial_start"):
            with patch("orchard.optimization.objective.objective.get_optimizer"):
                with patch("orchard.optimization.objective.objective.get_scheduler"):
                    with patch("orchard.optimization.objective.objective.get_criterion"):
                        result = objective(mock_trial)

    assert result == pytest.approx(0.95, abs=1e-5)
    mock_tracker.start_optuna_trial.assert_called_once_with(
        0, {"lr": 0.01, "pretrained": mock_cfg.architecture.pretrained}
    )
    mock_tracker.end_optuna_trial.assert_called_once()
