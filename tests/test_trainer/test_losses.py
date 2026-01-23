"""
Test Suite for Custom Loss Functions Module.

Tests FocalLoss implementation and helper functions.
"""

# Third-Party Imports
import pytest
import torch
import torch.nn as nn

# Internal Imports
from orchard.trainer.losses import FocalLoss, get_loss_name


#                    TESTS: FocalLoss                                         #
@pytest.mark.unit
def test_focal_loss_init_default():
    """Test FocalLoss initialization with default parameters."""
    loss_fn = FocalLoss()

    assert loss_fn.gamma == 2.0
    assert loss_fn.alpha == 1.0
    assert loss_fn.weight is None


@pytest.mark.unit
def test_focal_loss_init_custom():
    """Test FocalLoss initialization with custom parameters."""
    weights = torch.tensor([1.0, 2.0, 3.0])
    loss_fn = FocalLoss(gamma=3.0, alpha=0.5, weight=weights)

    assert loss_fn.gamma == 3.0
    assert loss_fn.alpha == 0.5
    assert torch.equal(loss_fn.weight, weights)


@pytest.mark.unit
def test_focal_loss_forward_basic():
    """Test FocalLoss forward pass with simple inputs."""
    loss_fn = FocalLoss()

    # Create simple logits and targets
    inputs = torch.tensor([[2.0, 1.0, 0.5], [1.0, 3.0, 0.5], [0.5, 1.0, 2.5]])
    targets = torch.tensor([0, 1, 2])

    loss = loss_fn(inputs, targets)

    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # Scalar
    assert loss.item() >= 0.0


@pytest.mark.unit
def test_focal_loss_forward_perfect_predictions():
    """Test FocalLoss with perfect predictions (should give very low loss)."""
    loss_fn = FocalLoss()

    # Perfect predictions: high logit for correct class
    inputs = torch.tensor([[10.0, -10.0, -10.0], [-10.0, 10.0, -10.0], [-10.0, -10.0, 10.0]])
    targets = torch.tensor([0, 1, 2])

    loss = loss_fn(inputs, targets)

    # Loss should be very small for perfect predictions
    assert loss.item() < 0.1


@pytest.mark.unit
def test_focal_loss_forward_poor_predictions():
    """Test FocalLoss with poor predictions (should give higher loss)."""
    loss_fn = FocalLoss()

    # Poor predictions: low logit for correct class
    inputs = torch.tensor([[-5.0, 5.0, 5.0], [5.0, -5.0, 5.0], [5.0, 5.0, -5.0]])
    targets = torch.tensor([0, 1, 2])

    loss = loss_fn(inputs, targets)

    # Loss should be higher for poor predictions
    assert loss.item() > 0.5


@pytest.mark.unit
def test_focal_loss_with_class_weights():
    """Test FocalLoss with class weights."""
    weights = torch.tensor([1.0, 2.0, 3.0])
    loss_fn = FocalLoss(weight=weights)

    inputs = torch.tensor([[2.0, 1.0, 0.5], [1.0, 3.0, 0.5], [0.5, 1.0, 2.5]])
    targets = torch.tensor([0, 1, 2])

    loss = loss_fn(inputs, targets)

    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0.0


@pytest.mark.unit
def test_focal_loss_gamma_effect():
    """Test that higher gamma reduces loss for well-classified examples."""
    inputs = torch.tensor([[5.0, 1.0, 0.5], [1.0, 6.0, 0.5], [0.5, 1.0, 7.0]])
    targets = torch.tensor([0, 1, 2])

    # Low gamma
    loss_fn_low = FocalLoss(gamma=0.5)
    loss_low = loss_fn_low(inputs, targets)

    # High gamma
    loss_fn_high = FocalLoss(gamma=5.0)
    loss_high = loss_fn_high(inputs, targets)

    # Higher gamma should give lower loss for well-classified examples
    assert loss_high.item() < loss_low.item()


@pytest.mark.unit
def test_focal_loss_alpha_effect():
    """Test that alpha parameter scales the loss."""
    inputs = torch.tensor([[2.0, 1.0, 0.5], [1.0, 3.0, 0.5], [0.5, 1.0, 2.5]])
    targets = torch.tensor([0, 1, 2])

    loss_fn_alpha1 = FocalLoss(alpha=1.0)
    loss_alpha1 = loss_fn_alpha1(inputs, targets)

    loss_fn_alpha2 = FocalLoss(alpha=2.0)
    loss_alpha2 = loss_fn_alpha2(inputs, targets)

    # Alpha=2 should give approximately 2x the loss
    assert abs(loss_alpha2.item() - 2 * loss_alpha1.item()) < 0.01


@pytest.mark.unit
def test_focal_loss_batch_size():
    """Test FocalLoss with different batch sizes."""
    loss_fn = FocalLoss()

    # Small batch
    inputs_small = torch.randn(4, 10)
    targets_small = torch.randint(0, 10, (4,))
    loss_small = loss_fn(inputs_small, targets_small)

    # Large batch
    inputs_large = torch.randn(64, 10)
    targets_large = torch.randint(0, 10, (64,))
    loss_large = loss_fn(inputs_large, targets_large)

    # Both should return scalar losses
    assert loss_small.dim() == 0
    assert loss_large.dim() == 0


@pytest.mark.unit
def test_focal_loss_many_classes():
    """Test FocalLoss with many classes."""
    loss_fn = FocalLoss()

    num_classes = 100
    batch_size = 32

    inputs = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))

    loss = loss_fn(inputs, targets)

    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert loss.item() >= 0.0


@pytest.mark.unit
def test_focal_loss_gradient_flow():
    """Test that FocalLoss allows gradient flow."""
    loss_fn = FocalLoss()

    inputs = torch.randn(8, 5, requires_grad=True)
    targets = torch.randint(0, 5, (8,))

    loss = loss_fn(inputs, targets)
    loss.backward()

    # Gradients should be computed
    assert inputs.grad is not None
    assert not torch.all(inputs.grad == 0)


@pytest.mark.unit
def test_focal_loss_is_module():
    """Test that FocalLoss is a proper nn.Module."""
    loss_fn = FocalLoss()

    assert isinstance(loss_fn, nn.Module)
    assert hasattr(loss_fn, "forward")


@pytest.mark.unit
def test_focal_loss_deterministic():
    """Test that FocalLoss gives same result for same inputs."""
    loss_fn = FocalLoss(gamma=2.0, alpha=1.0)

    inputs = torch.tensor([[2.0, 1.0, 0.5], [1.0, 3.0, 0.5]])
    targets = torch.tensor([0, 1])

    loss1 = loss_fn(inputs, targets)
    loss2 = loss_fn(inputs, targets)

    assert torch.equal(loss1, loss2)


#                    TESTS: get_loss_name                                     #
@pytest.mark.unit
def test_get_loss_name_focal_loss():
    """Test get_loss_name with FocalLoss."""
    loss_fn = FocalLoss()
    name = get_loss_name(loss_fn)

    assert name == "FocalLoss"


@pytest.mark.unit
def test_get_loss_name_cross_entropy():
    """Test get_loss_name with CrossEntropyLoss."""
    loss_fn = nn.CrossEntropyLoss()
    name = get_loss_name(loss_fn)

    assert name == "CrossEntropyLoss"


@pytest.mark.unit
def test_get_loss_name_mse():
    """Test get_loss_name with MSELoss."""
    loss_fn = nn.MSELoss()
    name = get_loss_name(loss_fn)

    assert name == "MSELoss"


@pytest.mark.unit
def test_get_loss_name_custom_class():
    """Test get_loss_name with custom loss class."""

    class CustomLoss(nn.Module):
        def forward(self, x, y):
            return (x - y).pow(2).mean()

    loss_fn = CustomLoss()
    name = get_loss_name(loss_fn)

    assert name == "CustomLoss"


@pytest.mark.unit
def test_get_loss_name_returns_string():
    """Test that get_loss_name always returns a string."""
    loss_fn = FocalLoss()
    name = get_loss_name(loss_fn)

    assert isinstance(name, str)
    assert len(name) > 0


# INTEGRATION TESTS
@pytest.mark.unit
def test_focal_loss_comparable_to_ce_when_gamma_zero():
    """Test that FocalLoss with gamma=0 behaves like CrossEntropy."""
    inputs = torch.randn(16, 10)
    targets = torch.randint(0, 10, (16,))

    # FocalLoss with gamma=0 and alpha=1
    focal_loss_fn = FocalLoss(gamma=0.0, alpha=1.0)
    focal_loss = focal_loss_fn(inputs, targets)

    # Standard Imports
    ce_loss_fn = nn.CrossEntropyLoss()
    ce_loss = ce_loss_fn(inputs, targets)

    # Should be very close (within numerical precision)
    assert abs(focal_loss.item() - ce_loss.item()) < 0.01


@pytest.mark.unit
def test_focal_loss_with_mixed_difficulty():
    """Test FocalLoss focuses on hard examples."""
    loss_fn = FocalLoss(gamma=2.0)

    # Mix of easy and hard examples
    inputs = torch.tensor(
        [
            [10.0, -5.0, -5.0],  # Easy: correct class has high logit
            [-5.0, -5.0, 10.0],  # Easy: correct class has high logit
            [1.0, 0.8, 0.9],  # Hard: all classes have similar logits
            [0.9, 1.0, 0.8],  # Hard: all classes have similar logits
        ]
    )
    targets = torch.tensor([0, 2, 0, 1])

    loss = loss_fn(inputs, targets)

    # Loss should be positive and influenced more by hard examples
    assert loss.item() > 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
