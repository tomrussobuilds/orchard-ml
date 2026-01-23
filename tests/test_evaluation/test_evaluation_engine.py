"""
Unit tests for the evaluation engine.
This module verifies model inference, Test-Time Augmentation (TTA) logic,
and logging behavior during the evaluation process.
"""

# Standard Imports
from unittest.mock import MagicMock, patch

# Third-Party Imports
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Internal Imports
from orchard.core import Config
from orchard.evaluation.engine import evaluate_model


# MOCK CLASSES
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)


# FIXTURES
@pytest.fixture
def mock_dataloader():
    """Create a dummy DataLoader with 2 batches of data."""
    x = torch.randn(4, 10)  # 4 total samples
    y = torch.tensor([0, 1, 0, 1])
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=2)


@pytest.fixture
def mock_config():
    """Create a minimal Config object."""
    cfg = MagicMock(spec=Config)
    return cfg


# TEST CASES
@pytest.mark.unit
@patch("orchard.evaluation.engine.compute_classification_metrics")
def test_evaluate_model_standard(mock_compute, mock_dataloader):
    """Test standard inference without TTA (Test-Time Augmentation)."""
    device = torch.device("cpu")
    model = SimpleModel()

    # Mock metrics to isolate the engine test
    mock_metrics = {"accuracy": 0.9, "auc": 0.95, "f1": 0.88}
    mock_compute.return_value = mock_metrics

    preds, labels, metrics, f1 = evaluate_model(model, mock_dataloader, device, use_tta=False)

    # Output assertions
    assert len(preds) == 4
    assert len(labels) == 4
    assert metrics["accuracy"] == 0.9
    assert f1 == 0.88

    # Verify model is set to evaluation mode
    assert not model.training
    # Verify compute_classification_metrics was called once
    mock_compute.assert_called_once()


@pytest.mark.unit
@patch("orchard.evaluation.engine.compute_classification_metrics")
@patch("orchard.evaluation.engine.adaptive_tta_predict")
def test_evaluate_model_with_tta(mock_tta, mock_compute, mock_dataloader, mock_config):
    """Test the execution path when TTA is enabled."""
    device = torch.device("cpu")
    model = SimpleModel()

    mock_compute.return_value = {"accuracy": 1.0, "auc": 1.0, "f1": 1.0}
    # Mock adaptive_tta_predict to return dummy probabilities (batch_size=2, classes=2)
    mock_tta.return_value = torch.tensor([[0.1, 0.9], [0.8, 0.2]])

    evaluate_model(model, mock_dataloader, device, use_tta=True, cfg=mock_config)

    # Verify TTA was called for each batch (2 batches in dataloader)
    assert mock_tta.call_count == 2


@pytest.mark.unit
@patch("orchard.evaluation.engine.compute_classification_metrics")
@patch("orchard.evaluation.engine.adaptive_tta_predict")
def test_tta_skipped_without_config(mock_tta, mock_compute, mock_dataloader):
    """Verify that TTA is skipped if cfg is None, even if use_tta is True."""
    device = torch.device("cpu")
    model = SimpleModel()
    mock_compute.return_value = {"accuracy": 0, "auc": 0, "f1": 0}

    evaluate_model(model, mock_dataloader, device, use_tta=True, cfg=None)

    # TTA should not be called because actual_tta becomes False
    mock_tta.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
