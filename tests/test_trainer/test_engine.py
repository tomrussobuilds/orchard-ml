"""
Test Suite for Training Engine.

Quick tests to cover core training/validation functions and eliminate codecov warnings.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from orchard.trainer.engine import mixup_data, train_one_epoch, validate_epoch


# FIXTURES
@pytest.fixture
def simple_model():
    """Simple 2-layer network for testing."""
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 10),
    )


@pytest.fixture
def simple_loader():
    """Mock dataloader with 2 batches."""
    batch1 = (torch.randn(4, 1, 28, 28), torch.randint(0, 10, (4,)))
    batch2 = (torch.randn(4, 1, 28, 28), torch.randint(0, 10, (4,)))
    loader = MagicMock()
    loader.__iter__ = MagicMock(return_value=iter([batch1, batch2]))
    loader.__len__ = MagicMock(return_value=2)
    return loader


@pytest.fixture
def criterion():
    """CrossEntropy loss."""
    return nn.CrossEntropyLoss()


@pytest.fixture
def optimizer(simple_model):
    """SGD optimizer."""
    return torch.optim.SGD(simple_model.parameters(), lr=0.01, momentum=0.0, weight_decay=0.0)


# TESTS: train_one_epoch
@pytest.mark.unit
def test_train_one_epoch_basic(simple_model, simple_loader, criterion, optimizer):
    """Test train_one_epoch completes without errors."""
    device = torch.device("cpu")
    loss = train_one_epoch(
        model=simple_model,
        loader=simple_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epoch=1,
        total_epochs=10,
        use_tqdm=False,
    )
    assert isinstance(loss, float)
    assert loss > 0


@pytest.mark.unit
def test_train_one_epoch_with_tqdm(simple_model, simple_loader, criterion, optimizer):
    """Test train_one_epoch with tqdm enabled."""
    device = torch.device("cpu")
    loss = train_one_epoch(
        model=simple_model,
        loader=simple_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epoch=1,
        total_epochs=10,
        use_tqdm=True,
    )
    assert isinstance(loss, float)
    assert loss > 0


@pytest.mark.unit
def test_train_one_epoch_with_grad_clip(simple_model, simple_loader, criterion, optimizer):
    """Test train_one_epoch with gradient clipping."""
    device = torch.device("cpu")
    loss = train_one_epoch(
        model=simple_model,
        loader=simple_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        grad_clip=1.0,
        use_tqdm=False,
    )
    assert isinstance(loss, float)


@pytest.mark.unit
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_train_one_epoch_scaler_grad_clip_coverage(
    simple_model, simple_loader, criterion, optimizer
):
    """Test scaler + grad_clip branch coverage."""
    device = torch.device("cpu")
    scaler = torch.amp.GradScaler(enabled=True)

    with patch("torch.nn.utils.clip_grad_norm_") as mock_clip:
        loss = train_one_epoch(
            model=simple_model,
            loader=simple_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            grad_clip=1.0,
            use_tqdm=False,
        )

        assert mock_clip.call_count >= 2
        assert isinstance(loss, float)


@pytest.mark.unit
@pytest.mark.filterwarnings(
    "ignore:torch.cuda.amp.GradScaler is enabled, but CUDA is not available:UserWarning"
)
def test_train_one_epoch_scaler_grad_clip_minimal():
    """Test scaler + grad_clip branch with minimal setup."""
    model = nn.Linear(10, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.0, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cpu")

    batch = (torch.randn(4, 10), torch.randint(0, 2, (4,)))
    loader = [batch]

    scaler = torch.amp.grad_scaler.GradScaler(enabled=True)

    original_unscale = scaler.unscale_
    unscale_called = [False]

    def tracked_unscale(optimizer):
        unscale_called[0] = True
        return original_unscale(optimizer)

    scaler.unscale_ = tracked_unscale

    loss = train_one_epoch(
        model=model,
        loader=loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scaler=scaler,
        grad_clip=1.0,
        use_tqdm=False,
    )

    assert unscale_called[0], "scaler.unscale_ should be called when grad_clip > 0"
    assert isinstance(loss, float)


@pytest.mark.unit
@pytest.mark.filterwarnings(
    "ignore:torch.cuda.amp.GradScaler is enabled, but CUDA is not available:UserWarning"
)
def test_train_one_epoch_with_scaler_and_grad_clip(
    simple_model, simple_loader, criterion, optimizer
):
    """Test train_one_epoch with AMP scaler AND gradient clipping."""
    device = torch.device("cpu")
    scaler = torch.amp.grad_scaler.GradScaler()

    loss = train_one_epoch(
        model=simple_model,
        loader=simple_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scaler=scaler,
        grad_clip=1.0,
        use_tqdm=False,
    )
    assert isinstance(loss, float)


@pytest.mark.unit
def test_train_one_epoch_with_mixup(simple_model, simple_loader, criterion, optimizer):
    """Test train_one_epoch with MixUp."""
    device = torch.device("cpu")

    def mock_mixup(x, y):
        return x, y, y, 0.5

    loss = train_one_epoch(
        model=simple_model,
        loader=simple_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        mixup_fn=mock_mixup,
        use_tqdm=False,
    )
    assert isinstance(loss, float)


@pytest.mark.unit
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for this branch")
def test_mixup_data_cuda_indexing():
    """Test mixup_data handles CUDA tensor indexing (lines 225-226)."""
    x = torch.randn(4, 3, 32, 32).cuda()
    y = torch.randint(0, 10, (4,)).cuda()

    mixed_x, _, _, _ = mixup_data(x, y, alpha=1.0)

    assert mixed_x.is_cuda
    assert mixed_x.device == x.device
    assert mixed_x.shape == x.shape


@pytest.mark.unit
def test_train_one_epoch_updates_tqdm_postfix(simple_model, simple_loader, criterion, optimizer):
    """Test that tqdm postfix is updated with loss."""
    device = torch.device("cpu")

    with patch("orchard.trainer.engine.tqdm") as mock_tqdm:
        mock_iterator = MagicMock()
        mock_tqdm.return_value = mock_iterator

        batch1 = (torch.randn(4, 1, 28, 28), torch.randint(0, 10, (4,)))
        batch2 = (torch.randn(4, 1, 28, 28), torch.randint(0, 10, (4,)))
        mock_iterator.__iter__ = MagicMock(return_value=iter([batch1, batch2]))

        _ = train_one_epoch(
            model=simple_model,
            loader=simple_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            use_tqdm=True,
        )

        assert mock_iterator.set_postfix.called


@pytest.mark.unit
def test_train_one_epoch_nan_loss_raises(simple_model, criterion, optimizer):
    """Test train_one_epoch raises RuntimeError when loss is NaN."""
    device = torch.device("cpu")

    # Create a model that produces NaN output by injecting NaN weights
    with torch.no_grad():
        for param in simple_model.parameters():
            param.fill_(float("nan"))

    batch = (torch.randn(4, 1, 28, 28), torch.randint(0, 10, (4,)))
    loader = MagicMock()
    loader.__iter__ = MagicMock(return_value=iter([batch]))
    loader.__len__ = MagicMock(return_value=1)

    with pytest.raises(RuntimeError, match="Training diverged"):
        train_one_epoch(
            model=simple_model,
            loader=loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            use_tqdm=False,
        )


# TESTS: VALIDATE EPOCH
@pytest.mark.unit
def test_validate_epoch_basic(simple_model, simple_loader, criterion):
    """Test validate_epoch returns correct metrics."""
    device = torch.device("cpu")
    metrics = validate_epoch(
        model=simple_model,
        val_loader=simple_loader,
        criterion=criterion,
        device=device,
    )

    assert isinstance(metrics, dict)
    assert "loss" in metrics
    assert "accuracy" in metrics
    assert "auc" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0


@pytest.mark.unit
def test_validate_epoch_binary_classification(simple_model, criterion):
    """Test validate_epoch with binary classification."""
    device = torch.device("cpu")

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 2),
    )

    batch = (torch.randn(8, 1, 28, 28), torch.randint(0, 2, (8,)))
    loader = MagicMock()
    loader.__iter__ = MagicMock(return_value=iter([batch]))

    metrics = validate_epoch(
        model=model,
        val_loader=loader,
        criterion=criterion,
        device=device,
    )

    assert isinstance(metrics, dict)
    assert "auc" in metrics


@pytest.mark.unit
def test_validate_epoch_auc_error_handling(simple_model, criterion):
    """Test validate_epoch handles AUC calculation errors."""
    device = torch.device("cpu")

    batch = (torch.randn(4, 1, 28, 28), torch.zeros(4, dtype=torch.long))
    loader = MagicMock()
    loader.__iter__ = MagicMock(return_value=iter([batch]))

    metrics = validate_epoch(
        model=simple_model,
        val_loader=loader,
        criterion=criterion,
        device=device,
    )

    assert metrics["auc"] >= 0.0


# TESTS: MIXUP DATA
@pytest.mark.unit
def test_mixup_data_basic():
    """Test mixup_data creates proper blends."""
    x = torch.randn(4, 3, 32, 32)
    y = torch.randint(0, 10, (4,))

    mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha=1.0)

    assert mixed_x.shape == x.shape
    assert y_a.shape == y.shape
    assert y_b.shape == y.shape
    assert 0.0 <= lam <= 1.0


@pytest.mark.unit
def test_mixup_data_disabled():
    """Test mixup_data with alpha=0 returns original data."""
    x = torch.randn(4, 3, 32, 32)
    y = torch.randint(0, 10, (4,))

    mixed_x, y_a, _, lam = mixup_data(x, y, alpha=0.0)

    assert torch.equal(mixed_x, x)
    assert torch.equal(y_a, y)
    assert lam == pytest.approx(1.0)


@pytest.mark.unit
def test_mixup_data_cuda_aware():
    """Test mixup_data handles CUDA tensors if available."""
    x = torch.randn(4, 3, 32, 32)
    y = torch.randint(0, 10, (4,))

    if torch.cuda.is_available():
        x = x.cuda()
        y = y.cuda()

    mixed_x, y_a, _, _ = mixup_data(x, y, alpha=1.0)

    assert mixed_x.device == x.device
    assert y_a.device == y.device


# TESTS: EMPTY LOADER GUARDS
@pytest.mark.unit
def test_train_one_epoch_empty_loader(simple_model, criterion, optimizer):
    """Test train_one_epoch handles empty loader gracefully."""
    device = torch.device("cpu")

    # Empty loader (returns no batches)
    empty_loader = MagicMock()
    empty_loader.__iter__ = MagicMock(return_value=iter([]))
    empty_loader.__len__ = MagicMock(return_value=0)

    loss = train_one_epoch(
        model=simple_model,
        loader=empty_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        use_tqdm=False,
    )

    assert loss == pytest.approx(0.0)


@pytest.mark.unit
def test_validate_epoch_empty_loader(simple_model, criterion):
    """Test validate_epoch handles empty loader gracefully."""
    device = torch.device("cpu")

    # Empty loader (returns no batches)
    empty_loader = MagicMock()
    empty_loader.__iter__ = MagicMock(return_value=iter([]))

    metrics = validate_epoch(
        model=simple_model,
        val_loader=empty_loader,
        criterion=criterion,
        device=device,
    )

    assert metrics == {"loss": 0.0, "accuracy": 0.0, "auc": 0.0, "f1": 0.0}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
