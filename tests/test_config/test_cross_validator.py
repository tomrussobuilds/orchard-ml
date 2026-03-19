"""
Test Suite for CrossDomainValidator.

Tests individual cross-domain validation checks in isolation.
End-to-end validation through Config is tested in test_manifest.py.
"""

from __future__ import annotations

from unittest.mock import MagicMock

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

    def test_mini_cnn_rejects_224(self) -> None:
        with pytest.raises(
            ValidationError,
            match=r"'mini_cnn' requires resolution \[28, 32, 64\]",
        ):
            Config(
                dataset=DatasetConfig(name="bloodmnist", resolution=224, force_rgb=True),
                architecture=ArchitectureConfig(name="mini_cnn", pretrained=False),
                hardware=HardwareConfig(device="cpu"),
            )

    def test_mini_cnn_accepts_32(self) -> None:
        cfg = Config(
            dataset=DatasetConfig(name="cifar10", resolution=32, force_rgb=True),
            architecture=ArchitectureConfig(name="mini_cnn", pretrained=False),
            hardware=HardwareConfig(device="cpu"),
        )
        assert cfg.dataset.resolution == 32

    def test_mini_cnn_accepts_64(self) -> None:
        cfg = Config(
            dataset=DatasetConfig(name="bloodmnist", resolution=64, force_rgb=True),
            architecture=ArchitectureConfig(name="mini_cnn", pretrained=False),
            hardware=HardwareConfig(device="cpu"),
        )
        assert cfg.dataset.resolution == 64

    def test_efficientnet_rejects_32(self) -> None:
        with pytest.raises(
            ValidationError,
            match="'efficientnet_b0' requires resolution=224",
        ):
            Config(
                dataset=DatasetConfig(name="cifar10", resolution=32, force_rgb=True),
                architecture=ArchitectureConfig(name="efficientnet_b0", pretrained=False),
                hardware=HardwareConfig(device="cpu"),
            )

    def test_efficientnet_rejects_64(self) -> None:
        with pytest.raises(
            ValidationError,
            match="'efficientnet_b0' requires resolution=224",
        ):
            Config(
                dataset=DatasetConfig(name="bloodmnist", resolution=64, force_rgb=True),
                architecture=ArchitectureConfig(name="efficientnet_b0", pretrained=False),
                hardware=HardwareConfig(device="cpu"),
            )

    def test_efficientnet_rejects_28(self) -> None:
        with pytest.raises(
            ValidationError,
            match="'efficientnet_b0' requires resolution=224",
        ):
            Config(
                dataset=DatasetConfig(name="bloodmnist", resolution=28),
                architecture=ArchitectureConfig(name="efficientnet_b0", pretrained=False),
                hardware=HardwareConfig(device="cpu"),
            )

    def test_resnet_18_accepts_28(self) -> None:
        cfg = Config(
            dataset=DatasetConfig(name="bloodmnist", resolution=28),
            architecture=ArchitectureConfig(name="resnet_18", pretrained=False),
            hardware=HardwareConfig(device="cpu"),
        )
        assert cfg.dataset.resolution == 28

    def test_resnet_18_accepts_64(self) -> None:
        cfg = Config(
            dataset=DatasetConfig(name="bloodmnist", resolution=64, force_rgb=True),
            architecture=ArchitectureConfig(name="resnet_18", pretrained=False),
            hardware=HardwareConfig(device="cpu"),
        )
        assert cfg.dataset.resolution == 64

    def test_resnet_18_accepts_32(self) -> None:
        cfg = Config(
            dataset=DatasetConfig(name="cifar10", resolution=32, force_rgb=True),
            architecture=ArchitectureConfig(name="resnet_18", pretrained=False),
            hardware=HardwareConfig(device="cpu"),
        )
        assert cfg.dataset.resolution == 32

    def test_resnet_18_accepts_224(self) -> None:
        cfg = Config(
            dataset=DatasetConfig(name="bloodmnist", resolution=224, force_rgb=True),
            architecture=ArchitectureConfig(name="resnet_18", pretrained=False),
            hardware=HardwareConfig(device="cpu"),
        )
        assert cfg.dataset.resolution == 224

    def test_resnet_18_rejects_112(self) -> None:
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

    def test_timm_model_bypasses_resolution_check(self) -> None:
        """timm/ models skip architecture-resolution validation."""
        cfg = Config(
            dataset=DatasetConfig(name="bloodmnist", resolution=224, force_rgb=True),
            architecture=ArchitectureConfig(name="timm/resnet10t", pretrained=False),
            hardware=HardwareConfig(device="cpu"),
        )
        assert cfg.architecture.name == "timm/resnet10t"

    def test_timm_model_accepts_any_resolution(self) -> None:
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

    def test_mixup_exceeds_total_raises(self) -> None:
        with pytest.raises(
            ValidationError,
            match="mixup_epochs .* exceeds total epochs",
        ):
            Config(
                training=TrainingConfig(epochs=5, mixup_epochs=10),
                hardware=HardwareConfig(device="cpu"),
            )

    def test_mixup_equal_to_total_passes(self) -> None:
        cfg = Config(
            training=TrainingConfig(epochs=10, mixup_epochs=10),
            hardware=HardwareConfig(device="cpu"),
        )
        assert cfg.training.mixup_epochs == 10


# AMP-DEVICE
@pytest.mark.unit
class TestCheckAmpDevice:
    """Tests for _check_amp_device."""

    def test_amp_on_cpu_auto_disabled(self) -> None:
        with pytest.warns(UserWarning, match="AMP.*CPU"):
            cfg = Config(
                training=TrainingConfig(use_amp=True),
                hardware=HardwareConfig(device="cpu"),
            )
        assert cfg.training.use_amp is False

    def test_amp_off_on_cpu_no_warning(self) -> None:
        cfg = Config(
            training=TrainingConfig(use_amp=False),
            hardware=HardwareConfig(device="cpu"),
        )
        assert cfg.training.use_amp is False


# PRETRAINED CHANNELS
@pytest.mark.unit
class TestCheckPretrainedChannels:
    """Tests for _check_pretrained_channels."""

    def test_pretrained_with_grayscale_raises(self, mock_grayscale_metadata: MagicMock) -> None:
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

    def test_pretrained_with_rgb_passes(self, mock_metadata_28: MagicMock) -> None:
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

    def test_min_lr_equal_to_lr_raises(self) -> None:
        with pytest.raises(ValidationError, match="min_lr"):
            Config(
                training=TrainingConfig(learning_rate=0.001, min_lr=0.001),
                hardware=HardwareConfig(device="cpu"),
            )

    def test_min_lr_greater_than_lr_raises(self) -> None:
        with pytest.raises(ValidationError, match="min_lr"):
            Config(
                training=TrainingConfig(learning_rate=0.001, min_lr=0.01),
                hardware=HardwareConfig(device="cpu"),
            )

    def test_min_lr_less_than_lr_passes(self) -> None:
        cfg = Config(
            training=TrainingConfig(learning_rate=0.01, min_lr=0.0001),
            hardware=HardwareConfig(device="cpu"),
        )
        assert cfg.training.min_lr < cfg.training.learning_rate


# CPU HIGH-RES PERFORMANCE
@pytest.mark.unit
class TestCheckCpuHighresPerformance:
    """Tests for _check_cpu_highres_performance."""

    def test_cpu_with_224_emits_warning(self) -> None:
        with pytest.warns(UserWarning, match="Training at resolution 224px on CPU"):
            Config(
                dataset=DatasetConfig(name="bloodmnist", resolution=224, force_rgb=True),
                architecture=ArchitectureConfig(name="resnet_18", pretrained=False),
                hardware=HardwareConfig(device="cpu"),
            )

    def test_cpu_with_28_no_warning(self) -> None:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            Config(
                dataset=DatasetConfig(name="bloodmnist", resolution=28),
                architecture=ArchitectureConfig(name="resnet_18", pretrained=False),
                training=TrainingConfig(use_amp=False),
                hardware=HardwareConfig(device="cpu"),
            )

    def test_cpu_with_64_no_warning(self) -> None:
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

    def test_max_samples_less_than_num_classes_raises(
        self, mock_metadata_many_classes: MagicMock
    ) -> None:
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

    def test_max_samples_sparse_emits_warning(self, mock_metadata_28: MagicMock) -> None:
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

    def test_max_samples_sufficient_no_warning(self, mock_metadata_28: MagicMock) -> None:
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

    def test_max_samples_none_no_check(self) -> None:
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

    def test_detection_skips_min_dataset_check(self, mock_metadata_many_classes: MagicMock) -> None:
        """Non-classification task_type should skip min dataset size check entirely."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            # Allow CPU+highres warning (not relevant to this test)
            warnings.filterwarnings("ignore", message="Training at resolution.*on CPU")
            # max_samples=30 < num_classes=50 would raise for classification
            Config(
                task_type="detection",
                dataset=DatasetConfig(
                    name="organamnist",
                    resolution=224,
                    metadata=mock_metadata_many_classes,
                    force_rgb=True,
                    max_samples=30,
                ),
                architecture=ArchitectureConfig(name="fasterrcnn", pretrained=False),
                training=TrainingConfig(use_amp=False, mixup_alpha=0.0),
                hardware=HardwareConfig(device="cpu"),
            )


# DIRECT VALIDATOR CALL
@pytest.mark.unit
class TestValidatorDirectCall:
    """Tests for CrossDomainValidator.validate() called directly."""

    def test_validate_returns_config(self) -> None:
        cfg = Config(hardware=HardwareConfig(device="cpu"))
        result = _CrossDomainValidator.validate(cfg)
        assert result is cfg


@pytest.mark.unit
class TestCheckDetectionConfig:
    """Tests for _check_detection_config validator."""

    def test_detection_with_classification_arch_raises(self) -> None:
        """Detection + classification architecture raises error."""
        with pytest.raises(ValidationError, match="not compatible.*detection"):
            Config(
                task_type="detection",
                dataset=DatasetConfig(name="organamnist", resolution=224, force_rgb=True),
                architecture=ArchitectureConfig(name="resnet_18", pretrained=False),
                training=TrainingConfig(use_amp=False, mixup_alpha=0.0),
                hardware=HardwareConfig(device="cpu"),
            )

    def test_detection_with_low_resolution_raises(self) -> None:
        """Detection + resolution < 224 raises error."""
        with pytest.raises(ValidationError, match="resolution >= 224"):
            Config(
                task_type="detection",
                dataset=DatasetConfig(name="organamnist", resolution=28, force_rgb=True),
                architecture=ArchitectureConfig(name="fasterrcnn", pretrained=False),
                training=TrainingConfig(use_amp=False, mixup_alpha=0.0),
                hardware=HardwareConfig(device="cpu"),
            )

    def test_detection_with_mixup_raises(self) -> None:
        """Detection + mixup_alpha > 0 raises error."""
        with pytest.raises(ValidationError, match="MixUp.*not compatible.*detection"):
            Config(
                task_type="detection",
                dataset=DatasetConfig(name="organamnist", resolution=224, force_rgb=True),
                architecture=ArchitectureConfig(name="fasterrcnn", pretrained=False),
                training=TrainingConfig(use_amp=False, mixup_alpha=0.4),
                hardware=HardwareConfig(device="cpu"),
            )

    def test_detection_valid_config_passes(self, mock_metadata_many_classes: MagicMock) -> None:
        """Valid detection config passes all checks."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            warnings.filterwarnings("ignore", message="Training at resolution.*on CPU")
            Config(
                task_type="detection",
                dataset=DatasetConfig(
                    name="organamnist",
                    resolution=224,
                    metadata=mock_metadata_many_classes,
                    force_rgb=True,
                ),
                architecture=ArchitectureConfig(name="fasterrcnn", pretrained=False),
                training=TrainingConfig(use_amp=False, mixup_alpha=0.0),
                hardware=HardwareConfig(device="cpu"),
            )

    def test_classification_skips_detection_check(self) -> None:
        """Classification task_type skips detection-specific checks entirely."""
        # resnet_18 is not a detection arch but should pass for classification
        Config(
            task_type="classification",
            architecture=ArchitectureConfig(name="resnet_18", pretrained=False),
            training=TrainingConfig(use_amp=False),
            hardware=HardwareConfig(device="cpu"),
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
