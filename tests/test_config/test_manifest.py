"""
Test Suite for Config Manifest.

Tests main Config class integration, cross-validation,
serialization, and from_recipe factory.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from orchard.core import (
    ArchitectureConfig,
    Config,
    DatasetConfig,
    HardwareConfig,
    TrainingConfig,
)
from orchard.core.config.manifest import (
    _MODELS_224_ONLY,
    _MODELS_LOW_RES,
    _RESOLUTIONS_224_ONLY,
    _RESOLUTIONS_LOW_RES,
    _deep_set,
)
from orchard.exceptions import OrchardConfigError


# CONFIG: BASIC CONSTRUCTION
@pytest.mark.unit
def test_config_defaults():
    """Test Config with all default sub-configs."""
    config = Config()

    assert config.hardware is not None
    assert config.training is not None
    assert config.dataset is not None
    assert config.architecture is not None


# CONFIG: CROSS-VALIDATION
@pytest.mark.unit
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_resnet_18_supports_all_resolutions(device):
    """
    resnet_18 supports 28x28, 64x64, and 224x224 resolutions.
    """
    config_28 = Config(
        dataset=DatasetConfig(name="bloodmnist", resolution=28),
        architecture=ArchitectureConfig(name="resnet_18", pretrained=False),
        training=TrainingConfig(),
        hardware=HardwareConfig(device=device),
    )
    assert config_28.dataset.resolution == 28

    config_64 = Config(
        dataset=DatasetConfig(name="bloodmnist", resolution=64, force_rgb=True),
        architecture=ArchitectureConfig(name="resnet_18", pretrained=False),
        training=TrainingConfig(),
        hardware=HardwareConfig(device=device),
    )
    assert config_64.dataset.resolution == 64

    config_224 = Config(
        dataset=DatasetConfig(name="bloodmnist", resolution=224, force_rgb=True),
        architecture=ArchitectureConfig(name="resnet_18", pretrained=False),
        training=TrainingConfig(),
        hardware=HardwareConfig(device=device),
    )
    assert config_224.dataset.resolution == 224


@pytest.mark.unit
@pytest.mark.parametrize("architecture_name", ["efficientnet_b0", "vit_tiny", "convnext_tiny"])
def test_224_models_require_resolution_224(architecture_name):
    """
    efficientnet_b0 and vit_tiny require 224x224 resolution.
    Using them with 28x28 should raise ValueError.
    """
    with pytest.raises(
        ValidationError,
        match=f"'{architecture_name}' requires resolution=224",
    ):
        Config(
            dataset=DatasetConfig(
                name="bloodmnist",
                resolution=28,
            ),
            architecture=ArchitectureConfig(
                name=architecture_name,
                pretrained=False,
            ),
            training=TrainingConfig(),
            hardware=HardwareConfig(device="cpu"),
        )


@pytest.mark.unit
def test_mini_cnn_rejects_224():
    """mini_cnn only supports 28x28 and 64x64 resolutions."""
    with pytest.raises(
        ValidationError,
        match=r"'mini_cnn' requires resolution \[28, 32, 64\]",
    ):
        Config(
            dataset=DatasetConfig(name="bloodmnist", resolution=224, force_rgb=True),
            architecture=ArchitectureConfig(name="mini_cnn", pretrained=False),
            training=TrainingConfig(),
            hardware=HardwareConfig(device="cpu"),
        )


@pytest.mark.unit
def test_mini_cnn_accepts_32():
    """mini_cnn accepts 32x32 resolution (CIFAR)."""
    cfg = Config(
        dataset=DatasetConfig(name="cifar10", resolution=32, force_rgb=True),
        architecture=ArchitectureConfig(name="mini_cnn", pretrained=False),
        training=TrainingConfig(),
        hardware=HardwareConfig(device="cpu"),
    )
    assert cfg.dataset.resolution == 32


@pytest.mark.unit
def test_resnet_18_accepts_32():
    """resnet_18 accepts 32x32 resolution (CIFAR)."""
    cfg = Config(
        dataset=DatasetConfig(name="cifar10", resolution=32, force_rgb=True),
        architecture=ArchitectureConfig(name="resnet_18", pretrained=False),
        training=TrainingConfig(),
        hardware=HardwareConfig(device="cpu"),
    )
    assert cfg.dataset.resolution == 32


@pytest.mark.unit
def test_resnet_18_rejects_invalid_resolution():
    """Unsupported resolution is caught by DatasetConfig.validate_resolution."""
    with pytest.raises(
        ValidationError,
        match=r"resolution=112 is not supported",
    ):
        Config(
            dataset=DatasetConfig(name="bloodmnist", resolution=112),
            architecture=ArchitectureConfig(name="resnet_18", pretrained=False),
            training=TrainingConfig(),
            hardware=HardwareConfig(device="cpu"),
        )


@pytest.mark.unit
def test_mixup_epochs_cannot_exceed_total_epochs_direct():
    """
    MixUp scheduling cannot exceed total training epochs.
    """
    with pytest.raises(
        ValidationError,
        match="mixup_epochs .* exceeds total epochs",
    ):
        Config(
            training=TrainingConfig(
                epochs=5,
                mixup_epochs=10,
            ),
            dataset=DatasetConfig(),
            architecture=ArchitectureConfig(),
            hardware=HardwareConfig(device="cpu"),
        )


@pytest.mark.unit
def test_amp_auto_disabled_on_cpu():
    """Test AMP is automatically disabled on CPU with warning."""
    with pytest.warns(UserWarning, match="AMP.*CPU"):
        cfg = Config(
            dataset=DatasetConfig(name="bloodmnist", resolution=28),
            architecture=ArchitectureConfig(name="mini_cnn", pretrained=False),
            training=TrainingConfig(use_amp=True),
            hardware=HardwareConfig(device="cpu"),
        )

    assert cfg.training.use_amp is False


@pytest.mark.unit
def test_pretrained_requires_rgb():
    """Test pretrained model validation enforces RGB channels."""
    with pytest.raises(ValidationError, match="Pretrained.*requires RGB"):
        Config(
            dataset=DatasetConfig(name="organcmnist", resolution=28, force_rgb=False),
            architecture=ArchitectureConfig(name="resnet_18", pretrained=True),
            training=TrainingConfig(),
            hardware=HardwareConfig(device="cpu"),
        )


@pytest.mark.unit
@pytest.mark.filterwarnings("ignore:AMP requires GPU.*:UserWarning")
def test_min_lr_equals_lr_direct_instantiation(mock_metadata_28):
    """Test min_lr == learning_rate validation via direct instantiation."""
    with pytest.raises(ValidationError):
        Config(
            dataset=DatasetConfig(
                name="bloodmnist",
                resolution=28,
                metadata=mock_metadata_28,
            ),
            architecture=ArchitectureConfig(name="mini_cnn", pretrained=False),
            training=TrainingConfig(
                learning_rate=0.001,
                min_lr=0.001,
            ),
            hardware=HardwareConfig(device="cpu"),
        )


# CONFIG: SERIALIZATION
@pytest.mark.unit
def test_dump_portable_converts_paths():
    """Test dump_portable() makes paths relative."""
    config = Config()

    portable = config.dump_portable()

    assert "dataset" in portable
    assert "telemetry" in portable


@pytest.mark.unit
def test_dump_serialized_json_compatible():
    """Test dump_serialized() produces JSON-compatible dict."""
    config = Config()

    serialized = config.dump_serialized()

    assert isinstance(serialized, dict)
    assert "hardware" in serialized
    assert "training" in serialized


# CONFIG: PROPERTIES
@pytest.mark.unit
def test_run_slug_property():
    """Test run_slug combines dataset and model names."""
    config = Config()

    slug = config.run_slug

    assert "bloodmnist" in slug
    assert config.architecture.name in slug


@pytest.mark.unit
def test_run_slug_sanitizes_timm_slash():
    """Test run_slug replaces / with _ for timm model names."""
    config = Config(
        architecture=ArchitectureConfig(name="timm/convnext_base", pretrained=False),
        dataset=DatasetConfig(name="bloodmnist", resolution=224, force_rgb=True),
        hardware=HardwareConfig(device="cpu"),
    )

    slug = config.run_slug

    assert "/" not in slug
    assert "timm_convnext_base" in slug


@pytest.mark.unit
def test_num_workers_property():
    """Test num_workers delegates to hardware config."""
    config = Config()

    workers = config.num_workers

    assert workers >= 0
    assert workers == config.hardware.effective_num_workers


# CONFIG: EDGE CASES
@pytest.mark.unit
def test_frozen_immutability():
    """Test Config is frozen (immutable)."""
    config = Config()

    with pytest.raises(ValidationError):
        config.training = None


@pytest.mark.unit
def test_min_lr_boundary_condition_line_106(mock_metadata_28):
    """
    Lines 106-110: msg creation and raise ValueError(msg) for min_lr >= learning_rate
    """
    with pytest.raises(ValidationError):
        Config(
            dataset=DatasetConfig(name="bloodmnist", resolution=28, metadata=mock_metadata_28),
            architecture=ArchitectureConfig(name="mini_cnn", pretrained=False),
            training=TrainingConfig(
                epochs=100, mixup_epochs=0, use_amp=False, learning_rate=0.001, min_lr=0.001
            ),
            hardware=HardwareConfig(device="cpu"),
        )

    with pytest.raises(ValidationError):
        Config(
            dataset=DatasetConfig(name="bloodmnist", resolution=28, metadata=mock_metadata_28),
            architecture=ArchitectureConfig(name="mini_cnn", pretrained=False),
            training=TrainingConfig(
                epochs=100, mixup_epochs=0, use_amp=False, learning_rate=0.001, min_lr=0.002
            ),
            hardware=HardwareConfig(device="cpu"),
        )


# ---------------------------------------------------------------------------
# Mutation-killing tests
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_models_low_res_exact_contents():
    """Assert _MODELS_LOW_RES contains exactly mini_cnn."""
    assert _MODELS_LOW_RES == frozenset({"mini_cnn"})
    assert len(_MODELS_LOW_RES) == 1


@pytest.mark.unit
def test_models_224_only_exact_contents():
    """Assert _MODELS_224_ONLY contains exactly the 3 expected models."""
    assert _MODELS_224_ONLY == frozenset({"efficientnet_b0", "vit_tiny", "convnext_tiny"})
    assert len(_MODELS_224_ONLY) == 3


@pytest.mark.unit
def test_resolutions_low_res_exact_contents():
    """Assert _RESOLUTIONS_LOW_RES contains exactly {28, 32, 64}."""
    assert _RESOLUTIONS_LOW_RES == frozenset({28, 32, 64})
    assert 28 in _RESOLUTIONS_LOW_RES
    assert 32 in _RESOLUTIONS_LOW_RES
    assert 64 in _RESOLUTIONS_LOW_RES
    assert 224 not in _RESOLUTIONS_LOW_RES


@pytest.mark.unit
def test_resolutions_224_only_exact_contents():
    """Assert _RESOLUTIONS_224_ONLY contains exactly {224}."""
    assert _RESOLUTIONS_224_ONLY == frozenset({224})
    assert 28 not in _RESOLUTIONS_224_ONLY


@pytest.mark.unit
def test_deep_set_nested_path():
    """Test _deep_set creates intermediate dicts for nested dot paths."""
    data: dict = {}
    _deep_set(data, "a.b.c", 42)
    assert data == {"a": {"b": {"c": 42}}}


@pytest.mark.unit
def test_deep_set_single_key():
    """Test _deep_set with a single key (no dots)."""
    data: dict = {}
    _deep_set(data, "key", "value")
    assert data == {"key": "value"}


@pytest.mark.unit
def test_deep_set_overwrites_existing():
    """Test _deep_set overwrites existing value at path."""
    data = {"a": {"b": 10}}
    _deep_set(data, "a.b", 20)
    assert data["a"]["b"] == 20


@pytest.mark.unit
def test_deep_set_preserves_siblings():
    """Test _deep_set preserves sibling keys in intermediate dicts."""
    data = {"a": {"existing": 99}}
    _deep_set(data, "a.new_key", 42)
    assert data["a"]["existing"] == 99
    assert data["a"]["new_key"] == 42


@pytest.mark.unit
def test_dump_portable_relative_data_root():
    """Test dump_portable converts data_root inside PROJECT_ROOT to relative."""
    from orchard.core.config import manifest as manifest_mod

    original_root = manifest_mod.PROJECT_ROOT
    cfg = Config(
        dataset=DatasetConfig(
            name="bloodmnist",
            resolution=28,
            data_root=str(original_root / "data" / "bloodmnist"),
        ),
        hardware=HardwareConfig(device="cpu"),
    )
    portable = cfg.dump_portable()
    ds = portable.get("dataset", {})
    dr = ds.get("data_root")
    if dr is not None:
        assert dr.startswith("./") or not Path(dr).is_absolute()


@pytest.mark.unit
def test_dump_portable_no_data_root():
    """Test dump_portable handles None data_root without crashing."""
    cfg = Config(hardware=HardwareConfig(device="cpu"))
    portable = cfg.dump_portable()
    assert "dataset" in portable


@pytest.mark.unit
def test_dump_serialized_returns_dict():
    """Test dump_serialized produces JSON-compatible dict with mode='json'."""
    cfg = Config(hardware=HardwareConfig(device="cpu"))
    serialized = cfg.dump_serialized()
    assert isinstance(serialized, dict)
    for key in ("hardware", "training", "dataset"):
        assert key in serialized


@pytest.mark.unit
def test_run_slug_exact_format():
    """Test run_slug produces '{dataset}_{model}' format."""
    cfg = Config(
        dataset=DatasetConfig(name="bloodmnist", resolution=28),
        architecture=ArchitectureConfig(name="mini_cnn", pretrained=False),
        hardware=HardwareConfig(device="cpu"),
    )
    assert cfg.run_slug == "bloodmnist_mini_cnn"


@pytest.mark.unit
def test_num_workers_delegates_to_hardware():
    """Test num_workers property delegates to hardware.effective_num_workers."""
    cfg = Config(
        hardware=HardwareConfig(device="cpu", reproducible=True),
    )
    assert cfg.num_workers == cfg.hardware.effective_num_workers
    assert cfg.num_workers == 0  # reproducible forces 0


@pytest.mark.unit
def test_from_recipe_missing_dataset_name(tmp_path):
    """Test from_recipe raises OrchardConfigError when dataset.name is missing."""
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text("dataset:\n  resolution: 28\n")
    with pytest.raises(OrchardConfigError, match="must specify 'dataset.name'"):
        Config.from_recipe(recipe)


@pytest.mark.unit
def test_from_recipe_unknown_dataset(tmp_path):
    """Test from_recipe raises OrchardConfigError for unknown dataset."""
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text("dataset:\n  name: nonexistent_dataset_xyz\n  resolution: 28\n")
    with pytest.raises(OrchardConfigError, match="not found at resolution"):
        Config.from_recipe(recipe)


@pytest.mark.unit
def test_from_recipe_with_overrides(tmp_path):
    """Test from_recipe applies dot-notation overrides."""
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(
        "dataset:\n  name: bloodmnist\n  resolution: 28\n"
        "training:\n  epochs: 50\n  mixup_epochs: 0\n"
        "architecture:\n  name: mini_cnn\n  pretrained: false\n"
        "hardware:\n  device: cpu\n"
    )
    cfg = Config.from_recipe(recipe, overrides={"training.epochs": 10})
    assert cfg.training.epochs == 10


@pytest.mark.unit
def test_from_recipe_default_resolution_28(tmp_path):
    """Test from_recipe uses resolution=28 as default."""
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(
        "dataset:\n  name: bloodmnist\n"
        "architecture:\n  name: mini_cnn\n  pretrained: false\n"
        "hardware:\n  device: cpu\n"
    )
    cfg = Config.from_recipe(recipe)
    assert cfg.dataset.resolution == 28


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
