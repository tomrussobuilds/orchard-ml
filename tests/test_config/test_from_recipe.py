"""
Test Suite for Config.from_recipe() and _deep_set helper.

Tests the YAML-first factory path used by the ``orchard`` CLI,
including dot-notation override application.
"""

from __future__ import annotations

import pytest
import yaml

from orchard.core.config.manifest import Config, _deep_set


# _DEEP_SET HELPER
@pytest.mark.unit
class TestDeepSet:
    """Tests for the _deep_set module-level helper."""

    def test_single_key(self):
        data = {"a": 1}
        _deep_set(data, "a", 2)
        assert data == {"a": 2}

    def test_nested_key(self):
        data = {"training": {"epochs": 60}}
        _deep_set(data, "training.epochs", 20)
        assert data["training"]["epochs"] == 20

    def test_creates_intermediate_dicts(self):
        data = {}
        _deep_set(data, "a.b.c", 42)
        assert data == {"a": {"b": {"c": 42}}}

    def test_preserves_siblings(self):
        data = {"training": {"epochs": 60, "seed": 42}}
        _deep_set(data, "training.epochs", 20)
        assert data["training"]["epochs"] == 20
        assert data["training"]["seed"] == 42

    def test_deep_three_levels(self):
        data = {"optuna": {"search_space": {"lr": 0.01}}}
        _deep_set(data, "optuna.search_space.lr", 0.001)
        assert data["optuna"]["search_space"]["lr"] == pytest.approx(0.001)

    def test_none_value(self):
        data = {"a": 1}
        _deep_set(data, "a", None)
        assert data["a"] is None

    def test_bool_value(self):
        data = {"training": {"use_amp": True}}
        _deep_set(data, "training.use_amp", False)
        assert data["training"]["use_amp"] is False


# CONFIG.FROM_RECIPE
@pytest.mark.integration
class TestFromRecipe:
    """Tests for Config.from_recipe() factory method."""

    def test_loads_valid_recipe(self, tmp_path):
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

    def test_applies_scalar_overrides(self, tmp_path):
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

    def test_override_dataset_name(self, tmp_path):
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

    def test_missing_dataset_name_raises(self, tmp_path):
        """from_recipe raises ValueError if dataset.name is missing."""
        yaml_content = {
            "architecture": {"name": "mini_cnn"},
            "training": {"epochs": 10},
        }
        recipe = tmp_path / "recipe.yaml"
        recipe.write_text(yaml.dump(yaml_content))

        with pytest.raises(ValueError, match="must specify 'dataset.name'"):
            Config.from_recipe(recipe)

    def test_unknown_dataset_raises(self, tmp_path):
        """from_recipe raises KeyError for unregistered datasets."""
        yaml_content = {
            "dataset": {"name": "nonexistent_dataset", "resolution": 28},
        }
        recipe = tmp_path / "recipe.yaml"
        recipe.write_text(yaml.dump(yaml_content))

        with pytest.raises(KeyError, match="nonexistent_dataset"):
            Config.from_recipe(recipe)

    def test_none_overrides_ignored(self, tmp_path):
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

    def test_empty_overrides_ignored(self, tmp_path):
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

    def test_loads_real_recipe(self):
        """from_recipe loads an actual recipe from the recipes/ directory."""
        from orchard.core.paths import PROJECT_ROOT

        recipe = PROJECT_ROOT / "recipes" / "config_mini_cnn.yaml"
        if not recipe.exists():
            pytest.skip("Recipe file not present")

        cfg = Config.from_recipe(recipe)

        assert cfg.dataset.dataset_name == "bloodmnist"
        assert cfg.architecture.name == "mini_cnn"

    def test_cross_validation_still_runs(self, tmp_path):
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
