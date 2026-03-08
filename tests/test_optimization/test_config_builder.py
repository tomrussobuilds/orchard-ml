"""
Mutation-killing tests for TrialConfigBuilder.

Targets surviving mutants in orchard/optimization/objective/config_builder.py
by asserting exact values, boundary conditions, and side-effects of build()
and _apply_param_overrides().
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from orchard.optimization.objective.config_builder import TrialConfigBuilder

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_builder(
    optuna_epochs: int = 10,
    resolution: int = 28,
    mixup_epochs: int = 5,
    training_epochs: int = 50,
) -> tuple[TrialConfigBuilder, MagicMock]:
    """Create a TrialConfigBuilder with a mock base config."""
    mock_cfg = MagicMock()
    mock_cfg.optuna.epochs = optuna_epochs
    sentinel_meta = {"num_classes": 5, "channel_count": 3}
    mock_cfg.dataset._ensure_metadata = sentinel_meta
    mock_cfg.dataset.resolution = resolution

    mock_cfg.model_dump.return_value = {
        "dataset": {"resolution": resolution, "metadata": None},
        "training": {"epochs": training_epochs, "mixup_epochs": mixup_epochs},
        "architecture": {},
        "augmentation": {},
    }

    return TrialConfigBuilder(mock_cfg), mock_cfg


# ---------------------------------------------------------------------------
# __init__ attribute storage
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_init_stores_base_cfg():
    """Assert base_cfg reference is stored."""
    builder, mock_cfg = _make_builder()
    assert builder.base_cfg is mock_cfg


@pytest.mark.unit
def test_init_stores_optuna_epochs():
    """Assert optuna_epochs is read from base_cfg.optuna.epochs."""
    builder, _ = _make_builder(optuna_epochs=42)
    assert builder.optuna_epochs == 42


@pytest.mark.unit
def test_init_stores_base_metadata():
    """Assert base_metadata is read from dataset._ensure_metadata."""
    builder, mock_cfg = _make_builder()
    assert builder.base_metadata is mock_cfg.dataset._ensure_metadata


# ---------------------------------------------------------------------------
# build() — metadata injection
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_build_injects_metadata_into_config():
    """Assert build() injects base_metadata into the dataset dict."""
    builder, mock_cfg = _make_builder()
    sentinel_meta = mock_cfg.dataset._ensure_metadata

    with patch("orchard.optimization.objective.config_builder.Config") as MockConfig:
        builder.build({})
        call_kwargs = MockConfig.call_args[1]
        assert call_kwargs["dataset"]["metadata"] is sentinel_meta


# ---------------------------------------------------------------------------
# build() — epochs override
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_build_overrides_epochs_to_optuna_epochs():
    """Assert build() sets training.epochs to optuna_epochs."""
    builder, _ = _make_builder(optuna_epochs=7, training_epochs=100)

    with patch("orchard.optimization.objective.config_builder.Config") as MockConfig:
        builder.build({})
        call_kwargs = MockConfig.call_args[1]
        assert call_kwargs["training"]["epochs"] == 7


# ---------------------------------------------------------------------------
# build() — mixup_epochs capping
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_build_caps_mixup_epochs_when_exceeds_optuna():
    """Assert mixup_epochs is capped to min(original, optuna_epochs)."""
    builder, _ = _make_builder(optuna_epochs=5, mixup_epochs=20)

    with patch("orchard.optimization.objective.config_builder.Config") as MockConfig:
        builder.build({})
        call_kwargs = MockConfig.call_args[1]
        assert call_kwargs["training"]["mixup_epochs"] == 5


@pytest.mark.unit
def test_build_keeps_mixup_epochs_when_already_lower():
    """Assert mixup_epochs stays when already <= optuna_epochs."""
    builder, _ = _make_builder(optuna_epochs=15, mixup_epochs=3)

    with patch("orchard.optimization.objective.config_builder.Config") as MockConfig:
        builder.build({})
        call_kwargs = MockConfig.call_args[1]
        assert call_kwargs["training"]["mixup_epochs"] == 3


@pytest.mark.unit
def test_build_mixup_equals_optuna_epochs():
    """Boundary: mixup_epochs == optuna_epochs → stays equal."""
    builder, _ = _make_builder(optuna_epochs=10, mixup_epochs=10)

    with patch("orchard.optimization.objective.config_builder.Config") as MockConfig:
        builder.build({})
        call_kwargs = MockConfig.call_args[1]
        assert call_kwargs["training"]["mixup_epochs"] == 10


# ---------------------------------------------------------------------------
# build() — resolution preservation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_build_preserves_resolution_when_none():
    """Assert resolution is set from base_cfg when model_dump returns None."""
    mock_cfg = MagicMock()
    mock_cfg.optuna.epochs = 10
    mock_cfg.dataset._ensure_metadata = {}
    mock_cfg.dataset.resolution = 224

    mock_cfg.model_dump.return_value = {
        "dataset": {"resolution": None},
        "training": {"epochs": 50, "mixup_epochs": 5},
        "architecture": {},
        "augmentation": {},
    }

    builder = TrialConfigBuilder(mock_cfg)

    with patch("orchard.optimization.objective.config_builder.Config") as MockConfig:
        builder.build({})
        call_kwargs = MockConfig.call_args[1]
        assert call_kwargs["dataset"]["resolution"] == 224


@pytest.mark.unit
def test_build_keeps_existing_resolution():
    """Assert resolution is NOT overridden when already present."""
    builder, _ = _make_builder(resolution=28)

    with patch("orchard.optimization.objective.config_builder.Config") as MockConfig:
        builder.build({})
        call_kwargs = MockConfig.call_args[1]
        assert call_kwargs["dataset"]["resolution"] == 28


# ---------------------------------------------------------------------------
# build() — returns Config instance
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_build_returns_config_instance():
    """Assert build() returns the Config constructor result."""
    builder, _ = _make_builder()
    sentinel = object()

    with patch(
        "orchard.optimization.objective.config_builder.Config", return_value=sentinel
    ) as MockConfig:
        result = builder.build({})
        MockConfig.assert_called_once()
        assert result is sentinel


# ---------------------------------------------------------------------------
# _apply_param_overrides — standard mapping
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_apply_overrides_maps_to_correct_sections():
    """Assert params are placed in their correct config sections."""
    builder, _ = _make_builder()

    config_dict = {
        "training": {"learning_rate": 0.01},
        "architecture": {"dropout": 0.0},
        "augmentation": {"rotation_angle": 0},
    }

    builder._apply_param_overrides(
        config_dict,
        {"learning_rate": 0.005, "dropout": 0.5, "rotation_angle": 30},
    )

    assert config_dict["training"]["learning_rate"] == 0.005
    assert config_dict["architecture"]["dropout"] == 0.5
    assert config_dict["augmentation"]["rotation_angle"] == 30


# ---------------------------------------------------------------------------
# _apply_param_overrides — SPECIAL_PARAMS
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_apply_overrides_special_param_model_name():
    """Assert model_name maps to architecture.name via SPECIAL_PARAMS."""
    builder, _ = _make_builder()

    config_dict = {"training": {}, "architecture": {"name": "old"}, "augmentation": {}}
    builder._apply_param_overrides(config_dict, {"model_name": "resnet_18"})

    assert config_dict["architecture"]["name"] == "resnet_18"


# ---------------------------------------------------------------------------
# _apply_param_overrides — weight_variant skip
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_apply_overrides_skips_none_weight_variant():
    """Assert weight_variant=None is skipped (non-ViT models)."""
    builder, _ = _make_builder()

    config_dict = {
        "training": {},
        "architecture": {"weight_variant": "original"},
        "augmentation": {},
    }
    builder._apply_param_overrides(config_dict, {"weight_variant": None})

    assert config_dict["architecture"]["weight_variant"] == "original"


@pytest.mark.unit
def test_apply_overrides_applies_non_none_weight_variant():
    """Assert weight_variant with a value IS applied."""
    builder, _ = _make_builder()

    config_dict = {
        "training": {},
        "architecture": {"weight_variant": "old"},
        "augmentation": {},
    }
    builder._apply_param_overrides(config_dict, {"weight_variant": "vit_tiny_patch16_224"})

    assert config_dict["architecture"]["weight_variant"] == "vit_tiny_patch16_224"


# ---------------------------------------------------------------------------
# _apply_param_overrides — unknown param
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_apply_overrides_unknown_param_not_added():
    """Unknown param not in any mapping is silently skipped."""
    builder, _ = _make_builder()

    config_dict = {"training": {}, "architecture": {}, "augmentation": {}}
    builder._apply_param_overrides(config_dict, {"totally_unknown": 99})

    assert "totally_unknown" not in config_dict["training"]
    assert "totally_unknown" not in config_dict["architecture"]
    assert "totally_unknown" not in config_dict["augmentation"]


# ---------------------------------------------------------------------------
# _apply_param_overrides — break after match
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_apply_overrides_does_not_duplicate_across_sections():
    """Assert a param is placed in exactly one section (break works)."""
    builder, _ = _make_builder()

    config_dict = {"training": {}, "architecture": {}, "augmentation": {}}
    builder._apply_param_overrides(config_dict, {"learning_rate": 0.001})

    # learning_rate belongs to "training" only
    assert config_dict["training"]["learning_rate"] == 0.001
    assert "learning_rate" not in config_dict["architecture"]
    assert "learning_rate" not in config_dict["augmentation"]
