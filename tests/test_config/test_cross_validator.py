"""
Test Suite for CrossDomainValidator.

Tests individual cross-domain validation checks in isolation.
End-to-end validation through Config is tested in test_manifest.py.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from orchard.core.config import (
    ArchitectureConfig,
    Config,
    DatasetConfig,
    HardwareConfig,
    TrainingConfig,
)
from orchard.core.config.manifest import _CrossDomainValidator


# ARCHITECTURE-RESOLUTION
@pytest.mark.unit
class TestCheckArchitectureResolution:
    """Tests for _check_architecture_resolution."""

    def test_mini_cnn_rejects_224(self):
        with pytest.raises(
            ValidationError,
            match=r"'mini_cnn' requires resolution \[28, 32, 64\]",
        ):
            Config(
                dataset=DatasetConfig(name="bloodmnist", resolution=224, force_rgb=True),
                architecture=ArchitectureConfig(name="mini_cnn", pretrained=False),
                hardware=HardwareConfig(device="cpu"),
            )

    def test_mini_cnn_accepts_32(self):
        cfg = Config(
            dataset=DatasetConfig(name="cifar10", resolution=32, force_rgb=True),
            architecture=ArchitectureConfig(name="mini_cnn", pretrained=False),
            hardware=HardwareConfig(device="cpu"),
        )
        assert cfg.dataset.resolution == 32

    def test_mini_cnn_accepts_64(self):
        cfg = Config(
            dataset=DatasetConfig(name="bloodmnist", resolution=64, force_rgb=True),
            architecture=ArchitectureConfig(name="mini_cnn", pretrained=False),
            hardware=HardwareConfig(device="cpu"),
        )
        assert cfg.dataset.resolution == 64

    def test_efficientnet_rejects_32(self):
        with pytest.raises(
            ValidationError,
            match="'efficientnet_b0' requires resolution=224",
        ):
            Config(
                dataset=DatasetConfig(name="cifar10", resolution=32, force_rgb=True),
                architecture=ArchitectureConfig(name="efficientnet_b0", pretrained=False),
                hardware=HardwareConfig(device="cpu"),
            )

    def test_efficientnet_rejects_64(self):
        with pytest.raises(
            ValidationError,
            match="'efficientnet_b0' requires resolution=224",
        ):
            Config(
                dataset=DatasetConfig(name="bloodmnist", resolution=64, force_rgb=True),
                architecture=ArchitectureConfig(name="efficientnet_b0", pretrained=False),
                hardware=HardwareConfig(device="cpu"),
            )

    def test_efficientnet_rejects_28(self):
        with pytest.raises(
            ValidationError,
            match="'efficientnet_b0' requires resolution=224",
        ):
            Config(
                dataset=DatasetConfig(name="bloodmnist", resolution=28),
                architecture=ArchitectureConfig(name="efficientnet_b0", pretrained=False),
                hardware=HardwareConfig(device="cpu"),
            )

    def test_resnet_18_accepts_28(self):
        cfg = Config(
            dataset=DatasetConfig(name="bloodmnist", resolution=28),
            architecture=ArchitectureConfig(name="resnet_18", pretrained=False),
            hardware=HardwareConfig(device="cpu"),
        )
        assert cfg.dataset.resolution == 28

    def test_resnet_18_accepts_64(self):
        cfg = Config(
            dataset=DatasetConfig(name="bloodmnist", resolution=64, force_rgb=True),
            architecture=ArchitectureConfig(name="resnet_18", pretrained=False),
            hardware=HardwareConfig(device="cpu"),
        )
        assert cfg.dataset.resolution == 64

    def test_resnet_18_accepts_32(self):
        cfg = Config(
            dataset=DatasetConfig(name="cifar10", resolution=32, force_rgb=True),
            architecture=ArchitectureConfig(name="resnet_18", pretrained=False),
            hardware=HardwareConfig(device="cpu"),
        )
        assert cfg.dataset.resolution == 32

    def test_resnet_18_accepts_224(self):
        cfg = Config(
            dataset=DatasetConfig(name="bloodmnist", resolution=224, force_rgb=True),
            architecture=ArchitectureConfig(name="resnet_18", pretrained=False),
            hardware=HardwareConfig(device="cpu"),
        )
        assert cfg.dataset.resolution == 224

    def test_resnet_18_rejects_112(self):
        """Unsupported resolution is now caught by DatasetConfig.validate_resolution."""
        with pytest.raises(
            ValidationError,
            match=r"resolution=112 is not supported",
        ):
            Config(
                dataset=DatasetConfig(name="bloodmnist", resolution=112),
                architecture=ArchitectureConfig(name="resnet_18", pretrained=False),
                hardware=HardwareConfig(device="cpu"),
            )

    def test_timm_model_bypasses_resolution_check(self):
        """timm/ models skip architecture-resolution validation."""
        cfg = Config(
            dataset=DatasetConfig(name="bloodmnist", resolution=224, force_rgb=True),
            architecture=ArchitectureConfig(name="timm/resnet10t", pretrained=False),
            hardware=HardwareConfig(device="cpu"),
        )
        assert cfg.architecture.name == "timm/resnet10t"

    def test_timm_model_accepts_any_resolution(self):
        """timm/ models accept resolutions that would fail for built-in models."""
        cfg = Config(
            dataset=DatasetConfig(name="bloodmnist", resolution=28),
            architecture=ArchitectureConfig(name="timm/resnet10t", pretrained=False),
            hardware=HardwareConfig(device="cpu"),
        )
        assert cfg.dataset.resolution == 28


# MIXUP EPOCHS
@pytest.mark.unit
class TestCheckMixupEpochs:
    """Tests for _check_mixup_epochs."""

    def test_mixup_exceeds_total_raises(self):
        with pytest.raises(
            ValidationError,
            match="mixup_epochs .* exceeds total epochs",
        ):
            Config(
                training=TrainingConfig(epochs=5, mixup_epochs=10),
                hardware=HardwareConfig(device="cpu"),
            )

    def test_mixup_equal_to_total_passes(self):
        cfg = Config(
            training=TrainingConfig(epochs=10, mixup_epochs=10),
            hardware=HardwareConfig(device="cpu"),
        )
        assert cfg.training.mixup_epochs == 10


# AMP-DEVICE
@pytest.mark.unit
class TestCheckAmpDevice:
    """Tests for _check_amp_device."""

    def test_amp_on_cpu_auto_disabled(self):
        with pytest.warns(UserWarning, match="AMP.*CPU"):
            cfg = Config(
                training=TrainingConfig(use_amp=True),
                hardware=HardwareConfig(device="cpu"),
            )
        assert cfg.training.use_amp is False

    def test_amp_off_on_cpu_no_warning(self):
        cfg = Config(
            training=TrainingConfig(use_amp=False),
            hardware=HardwareConfig(device="cpu"),
        )
        assert cfg.training.use_amp is False


# PRETRAINED CHANNELS
@pytest.mark.unit
class TestCheckPretrainedChannels:
    """Tests for _check_pretrained_channels."""

    def test_pretrained_with_grayscale_raises(self, mock_grayscale_metadata):
        with pytest.raises(
            ValidationError,
            match="Pretrained.*requires RGB",
        ):
            Config(
                dataset=DatasetConfig(
                    name="pneumoniamnist",
                    resolution=28,
                    metadata=mock_grayscale_metadata,
                    force_rgb=False,
                ),
                architecture=ArchitectureConfig(name="resnet_18", pretrained=True),
                hardware=HardwareConfig(device="cpu"),
            )

    def test_pretrained_with_rgb_passes(self, mock_metadata_28):
        cfg = Config(
            dataset=DatasetConfig(
                name="bloodmnist",
                resolution=28,
                metadata=mock_metadata_28,
            ),
            architecture=ArchitectureConfig(name="resnet_18", pretrained=True),
            hardware=HardwareConfig(device="cpu"),
        )
        assert cfg.architecture.pretrained is True


# LR BOUNDS
@pytest.mark.unit
class TestCheckLrBounds:
    """Tests for _check_lr_bounds."""

    def test_min_lr_equal_to_lr_raises(self):
        with pytest.raises(ValidationError, match="min_lr"):
            Config(
                training=TrainingConfig(learning_rate=0.001, min_lr=0.001),
                hardware=HardwareConfig(device="cpu"),
            )

    def test_min_lr_greater_than_lr_raises(self):
        with pytest.raises(ValidationError, match="min_lr"):
            Config(
                training=TrainingConfig(learning_rate=0.001, min_lr=0.01),
                hardware=HardwareConfig(device="cpu"),
            )

    def test_min_lr_less_than_lr_passes(self):
        cfg = Config(
            training=TrainingConfig(learning_rate=0.01, min_lr=0.0001),
            hardware=HardwareConfig(device="cpu"),
        )
        assert cfg.training.min_lr < cfg.training.learning_rate


# CPU HIGH-RES PERFORMANCE
@pytest.mark.unit
class TestCheckCpuHighresPerformance:
    """Tests for _check_cpu_highres_performance."""

    def test_cpu_with_224_emits_warning(self):
        with pytest.warns(UserWarning, match="Training at resolution 224px on CPU"):
            Config(
                dataset=DatasetConfig(name="bloodmnist", resolution=224, force_rgb=True),
                architecture=ArchitectureConfig(name="resnet_18", pretrained=False),
                hardware=HardwareConfig(device="cpu"),
            )

    def test_cpu_with_28_no_warning(self):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            Config(
                dataset=DatasetConfig(name="bloodmnist", resolution=28),
                architecture=ArchitectureConfig(name="resnet_18", pretrained=False),
                training=TrainingConfig(use_amp=False),
                hardware=HardwareConfig(device="cpu"),
            )

    def test_cpu_with_64_no_warning(self):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            Config(
                dataset=DatasetConfig(name="bloodmnist", resolution=64, force_rgb=True),
                architecture=ArchitectureConfig(name="mini_cnn", pretrained=False),
                training=TrainingConfig(use_amp=False),
                hardware=HardwareConfig(device="cpu"),
            )


# MIN DATASET SIZE
@pytest.mark.unit
class TestCheckMinDatasetSize:
    """Tests for _check_min_dataset_size."""

    def test_max_samples_less_than_num_classes_raises(self, mock_metadata_many_classes):
        """max_samples < num_classes (50) should raise ValueError."""
        with pytest.raises(ValidationError, match="must be >= num_classes"):
            Config(
                dataset=DatasetConfig(
                    name="organamnist",
                    resolution=28,
                    metadata=mock_metadata_many_classes,
                    force_rgb=True,
                    max_samples=30,
                ),
                architecture=ArchitectureConfig(name="resnet_18", pretrained=False),
                training=TrainingConfig(use_amp=False),
                hardware=HardwareConfig(device="cpu"),
            )

    def test_max_samples_sparse_emits_warning(self, mock_metadata_28):
        """max_samples < 10 * num_classes (8) should warn."""
        with pytest.warns(UserWarning, match="less than 10x num_classes"):
            Config(
                dataset=DatasetConfig(
                    name="bloodmnist",
                    resolution=28,
                    metadata=mock_metadata_28,
                    max_samples=50,
                ),
                architecture=ArchitectureConfig(name="resnet_18", pretrained=False),
                training=TrainingConfig(use_amp=False),
                hardware=HardwareConfig(device="cpu"),
            )

    def test_max_samples_sufficient_no_warning(self, mock_metadata_28):
        """max_samples >= 10 * num_classes (80) should not warn."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            Config(
                dataset=DatasetConfig(
                    name="bloodmnist",
                    resolution=28,
                    metadata=mock_metadata_28,
                    max_samples=100,
                ),
                architecture=ArchitectureConfig(name="resnet_18", pretrained=False),
                training=TrainingConfig(use_amp=False),
                hardware=HardwareConfig(device="cpu"),
            )

    def test_max_samples_none_no_check(self):
        """max_samples=None should skip validation entirely."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            Config(
                dataset=DatasetConfig(name="bloodmnist", resolution=28),
                architecture=ArchitectureConfig(name="resnet_18", pretrained=False),
                training=TrainingConfig(use_amp=False),
                hardware=HardwareConfig(device="cpu"),
            )


# DIRECT VALIDATOR CALL
@pytest.mark.unit
class TestValidatorDirectCall:
    """Tests for CrossDomainValidator.validate() called directly."""

    def test_validate_returns_config(self):
        cfg = Config(hardware=HardwareConfig(device="cpu"))
        result = _CrossDomainValidator.validate(cfg)
        assert result is cfg


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
