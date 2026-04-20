"""
Test Suite for Config.from_recipe() and _deep_set helper.

Tests the YAML-first factory path used by the ``orchard`` CLI,
including dot-notation override application.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from orchard.core.config.manifest import Config, _deep_set


# _DEEP_SET HELPER
@pytest.mark.unit
class TestDeepSet:
    """Tests for the _deep_set module-level helper."""

    def test_single_key(self) -> None:
        data = {"a": 1}
        _deep_set(data, "a", 2)
        assert data == {"a": 2}

    def test_nested_key(self) -> None:
        data = {"training": {"epochs": 60}}
        _deep_set(data, "training.epochs", 20)
        assert data["training"]["epochs"] == 20

    def test_creates_intermediate_dicts(self) -> None:
        data: dict[str, Any] = {}
        _deep_set(data, "a.b.c", 42)
        assert data == {"a": {"b": {"c": 42}}}

    def test_preserves_siblings(self) -> None:
        data = {"training": {"epochs": 60, "seed": 42}}
        _deep_set(data, "training.epochs", 20)
        assert data["training"]["epochs"] == 20
        assert data["training"]["seed"] == 42

    def test_deep_three_levels(self) -> None:
        data = {"optuna": {"search_space": {"lr": 0.01}}}
        _deep_set(data, "optuna.search_space.lr", 0.001)
        assert data["optuna"]["search_space"]["lr"] == pytest.approx(0.001)

    def test_none_value(self) -> None:
        data = {"a": 1}
        _deep_set(data, "a", None)
        assert data["a"] is None

    def test_bool_value(self) -> None:
        data = {"training": {"use_amp": True}}
        _deep_set(data, "training.use_amp", False)
        assert data["training"]["use_amp"] is False


# CONFIG.FROM_RECIPE
@pytest.mark.integration
class TestFromRecipe:
    """Tests for Config.from_recipe() factory method."""

    def test_loads_valid_recipe(self, tmp_path: Path) -> None:
        """from_recipe loads a minimal valid YAML recipe."""
        yaml_content = {
            "dataset": {"name": "bloodmnist", "resolution": 28, "force_rgb": True},
            "architecture": {"name": "mini_cnn", "pretrained": False},
            "training": {"epochs": 10, "mixup_epochs": 0, "use_amp": False},
            "hardware": {"device": "cpu"},
        }
        recipe = tmp_path / "recipe.yaml"
        recipe.write_text(yaml.dump(yaml_content))

        cfg = Config.from_recipe(recipe)

        assert cfg.dataset.dataset_name == "bloodmnist"
        assert cfg.architecture.name == "mini_cnn"
        assert cfg.training.epochs == 10

    def test_applies_scalar_overrides(self, tmp_path: Path) -> None:
        """from_recipe applies dot-notation overrides before instantiation."""
        yaml_content = {
            "dataset": {"name": "bloodmnist", "resolution": 28, "force_rgb": True},
            "architecture": {"name": "mini_cnn", "pretrained": False},
            "training": {"epochs": 60, "mixup_epochs": 0, "use_amp": False},
            "hardware": {"device": "cpu"},
        }
        recipe = tmp_path / "recipe.yaml"
        recipe.write_text(yaml.dump(yaml_content))

        cfg = Config.from_recipe(
            recipe,
            overrides={"training.epochs": 20, "training.seed": 123},
        )

        assert cfg.training.epochs == 20
        assert cfg.training.seed == 123

    def test_override_dataset_name(self, tmp_path: Path) -> None:
        """Overriding dataset.name re-resolves metadata correctly."""
        yaml_content = {
            "dataset": {"name": "bloodmnist", "resolution": 28, "force_rgb": True},
            "architecture": {"name": "mini_cnn", "pretrained": False},
            "training": {"epochs": 10, "mixup_epochs": 0, "use_amp": False},
            "hardware": {"device": "cpu"},
        }
        recipe = tmp_path / "recipe.yaml"
        recipe.write_text(yaml.dump(yaml_content))

        cfg = Config.from_recipe(
            recipe,
            overrides={"dataset.name": "pathmnist"},
        )

        assert cfg.dataset.dataset_name == "pathmnist"

    def test_missing_dataset_name_raises(self, tmp_path: Path) -> None:
        """from_recipe raises ValueError if dataset.name is missing."""
        yaml_content = {
            "architecture": {"name": "mini_cnn"},
            "training": {"epochs": 10},
        }
        recipe = tmp_path / "recipe.yaml"
        recipe.write_text(yaml.dump(yaml_content))

        with pytest.raises(ValueError, match="must specify 'dataset.name'"):
            Config.from_recipe(recipe)

    def test_unknown_dataset_raises(self, tmp_path: Path) -> None:
        """from_recipe raises KeyError for unregistered datasets."""
        yaml_content = {
            "dataset": {"name": "nonexistent_dataset", "resolution": 28},
        }
        recipe = tmp_path / "recipe.yaml"
        recipe.write_text(yaml.dump(yaml_content))

        with pytest.raises(ValueError, match="nonexistent_dataset"):
            Config.from_recipe(recipe)

    def test_none_overrides_ignored(self, tmp_path: Path) -> None:
        """from_recipe works when overrides is None."""
        yaml_content = {
            "dataset": {"name": "bloodmnist", "resolution": 28, "force_rgb": True},
            "architecture": {"name": "mini_cnn", "pretrained": False},
            "training": {"epochs": 10, "mixup_epochs": 0, "use_amp": False},
            "hardware": {"device": "cpu"},
        }
        recipe = tmp_path / "recipe.yaml"
        recipe.write_text(yaml.dump(yaml_content))

        cfg = Config.from_recipe(recipe, overrides=None)

        assert cfg.training.epochs == 10

    def test_empty_overrides_ignored(self, tmp_path: Path) -> None:
        """from_recipe works when overrides is an empty dict."""
        yaml_content = {
            "dataset": {"name": "bloodmnist", "resolution": 28, "force_rgb": True},
            "architecture": {"name": "mini_cnn", "pretrained": False},
            "training": {"epochs": 10, "mixup_epochs": 0, "use_amp": False},
            "hardware": {"device": "cpu"},
        }
        recipe = tmp_path / "recipe.yaml"
        recipe.write_text(yaml.dump(yaml_content))

        cfg = Config.from_recipe(recipe, overrides={})

        assert cfg.training.epochs == 10

    def test_loads_real_recipe(self) -> None:
        """from_recipe loads an actual recipe from the recipes/ directory."""
        from orchard.core.paths import PROJECT_ROOT

        recipe = PROJECT_ROOT / "recipes" / "config_mini_cnn.yaml"
        if not recipe.exists():
            pytest.skip("Recipe file not present")

        cfg = Config.from_recipe(recipe)

        assert cfg.dataset.dataset_name == "bloodmnist"
        assert cfg.architecture.name == "mini_cnn"

    def test_resolution_read_from_dataset_section(self, tmp_path: Path) -> None:
        """resolution key in dataset section overrides default 28."""
        yaml_content = {
            "dataset": {"name": "cifar10", "resolution": 32},
            "architecture": {"name": "mini_cnn", "pretrained": False},
            "training": {"epochs": 1, "mixup_epochs": 0, "use_amp": False},
            "hardware": {"device": "cpu"},
        }
        recipe = tmp_path / "recipe.yaml"
        recipe.write_text(yaml.dump(yaml_content))

        cfg = Config.from_recipe(recipe)

        # Kill mutants: dataset_section.get("resolution", 28) key → None / wrong string / UPPERCASE
        # Wrong key falls back to default 28 → cifar10 not in 28px registry → error
        assert cfg.dataset.resolution == 32
        assert cfg.dataset.metadata.native_resolution == 32

    def test_default_resolution_when_omitted(self, tmp_path: Path) -> None:
        """resolution defaults to 28 when absent from dataset section."""
        yaml_content = {
            "dataset": {"name": "bloodmnist", "force_rgb": True},
            "architecture": {"name": "mini_cnn", "pretrained": False},
            "training": {"epochs": 1, "mixup_epochs": 0, "use_amp": False},
            "hardware": {"device": "cpu"},
        }
        recipe = tmp_path / "recipe.yaml"
        recipe.write_text(yaml.dump(yaml_content))

        cfg = Config.from_recipe(recipe)

        # Kill mutant: default 28 → 29 causes get_registry(29) → unsupported resolution error
        assert cfg.dataset.resolution == 28

    def test_task_type_invalid_in_yaml_raises(self, tmp_path: Path) -> None:
        """task_type key in YAML is read; invalid value raises ValueError."""
        yaml_content = {
            "task_type": "segmentation",
            "dataset": {"name": "bloodmnist", "resolution": 28, "force_rgb": True},
            "architecture": {"name": "mini_cnn", "pretrained": False},
            "training": {"epochs": 1, "mixup_epochs": 0, "use_amp": False},
            "hardware": {"device": "cpu"},
        }
        recipe = tmp_path / "recipe.yaml"
        recipe.write_text(yaml.dump(yaml_content))

        # Kill mutants: key → None/TASK_TYPE/XXtask_typeXX all fall back to "classification" → no error
        # Kill mutant: get_registry(resolution,) drops task_type → default "classification" → no error
        with pytest.raises(ValueError, match="Unknown task_type"):
            Config.from_recipe(recipe)

    def test_task_type_default_classification_when_absent(self, tmp_path: Path) -> None:
        """from_recipe defaults task_type to 'classification' when absent."""
        yaml_content = {
            "dataset": {"name": "bloodmnist", "resolution": 28, "force_rgb": True},
            "architecture": {"name": "mini_cnn", "pretrained": False},
            "training": {"epochs": 1, "mixup_epochs": 0, "use_amp": False},
            "hardware": {"device": "cpu"},
        }
        recipe = tmp_path / "recipe.yaml"
        recipe.write_text(yaml.dump(yaml_content))

        cfg = Config.from_recipe(recipe)

        # Kill mutant: default "classification" → "XXclassificationXX" → get_registry raises ValueError
        assert cfg.task_type == "classification"

    def test_unknown_dataset_error_lists_available(self, tmp_path: Path) -> None:
        """Error for unknown dataset includes the available list in brackets."""
        yaml_content = {
            "dataset": {"name": "nonexistent_dataset", "resolution": 28},
        }
        recipe = tmp_path / "recipe.yaml"
        recipe.write_text(yaml.dump(yaml_content))

        with pytest.raises(ValueError) as exc_info:
            Config.from_recipe(recipe)

        msg = str(exc_info.value)
        assert "nonexistent_dataset" in msg
        # Kill mutant: available = None → message would say "None" not a real list
        assert "[" in msg and "]" in msg

    def test_metadata_injected_into_config(self, tmp_path: Path) -> None:
        """Metadata is properly fetched and injected into the config."""
        yaml_content = {
            "dataset": {"name": "bloodmnist", "resolution": 28, "force_rgb": True},
            "architecture": {"name": "mini_cnn", "pretrained": False},
            "training": {"epochs": 1, "mixup_epochs": 0, "use_amp": False},
            "hardware": {"device": "cpu"},
        }
        recipe = tmp_path / "recipe.yaml"
        recipe.write_text(yaml.dump(yaml_content))

        cfg = Config.from_recipe(recipe)

        # Kill mutants: metadata = None / setdefault("dataset", {})["metadata"] = None
        assert cfg.dataset.metadata is not None
        assert cfg.dataset.metadata.name == "bloodmnist"

    def test_optuna_preset_full_warns_on_full_extra_override(self, tmp_path: Path) -> None:
        """With search_space_preset 'full' (default), FULL_EXTRA overrides trigger warning."""
        yaml_content = {
            "dataset": {"name": "bloodmnist", "resolution": 28, "force_rgb": True},
            "architecture": {"name": "mini_cnn", "pretrained": False},
            "training": {"epochs": 1, "mixup_epochs": 0, "use_amp": False},
            "hardware": {"device": "cpu"},
            "optuna": {
                "study_name": "test",
                "n_trials": 1,
                "epochs": 10,
                "enable_early_stopping": False,
                "sampler_type": "tpe",
            },
        }
        recipe = tmp_path / "recipe.yaml"
        recipe.write_text(yaml.dump(yaml_content))

        # training.criterion_type is in FULL_EXTRA but not QUICK
        # Kill mutants: default "full" → "XXfullXX"/"FULL"/None → only QUICK keys → no warning
        with pytest.warns(UserWarning, match="will be ignored"):
            Config.from_recipe(recipe, overrides={"training.criterion_type": "focal"})

    def test_optuna_preset_quick_no_warn_on_full_extra_override(self, tmp_path: Path) -> None:
        """With search_space_preset 'quick', FULL_EXTRA-only overrides do not warn."""
        import warnings as _warnings

        yaml_content = {
            "dataset": {"name": "bloodmnist", "resolution": 28, "force_rgb": True},
            "architecture": {"name": "mini_cnn", "pretrained": False},
            "training": {"epochs": 10, "mixup_epochs": 0, "use_amp": False},
            "hardware": {"device": "cpu"},
            "optuna": {
                "study_name": "test",
                "n_trials": 1,
                "epochs": 10,
                "enable_early_stopping": False,
                "sampler_type": "tpe",
                "search_space_preset": "quick",
            },
        }
        recipe = tmp_path / "recipe.yaml"
        recipe.write_text(yaml.dump(yaml_content))

        # Kill mutants: key → None/UPPERCASE → reads "full" (default) → FULL_EXTRA keys → warning
        with _warnings.catch_warnings():
            _warnings.simplefilter("error", UserWarning)
            Config.from_recipe(recipe, overrides={"training.criterion_type": "focal"})

    def test_cross_validation_still_runs(self, tmp_path: Path) -> None:
        """from_recipe triggers cross-domain validation (e.g. resolution mismatch)."""
        yaml_content = {
            "dataset": {"name": "bloodmnist", "resolution": 224, "force_rgb": True},
            "architecture": {"name": "mini_cnn", "pretrained": False},
            "training": {"epochs": 10, "mixup_epochs": 0, "use_amp": False},
            "hardware": {"device": "cpu"},
        }
        recipe = tmp_path / "recipe.yaml"
        recipe.write_text(yaml.dump(yaml_content))

        with pytest.raises(Exception, match=r"mini_cnn.*requires resolution \[28, 32, 64\]"):
            Config.from_recipe(recipe)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
