"""
Test Suite for RunPaths Dynamic Directory Management.

Tests atomic run isolation, unique ID generation, directory creation,
and path resolution for experiment artifacts.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from orchard.core.paths import OUTPUTS_ROOT, RunPaths


# RUNPATHS: CLASS CONSTANTS
@pytest.mark.unit
def test_sub_dirs_constant():
    """Test SUB_DIRS class constant contains all required subdirectories."""
    assert hasattr(RunPaths, "SUB_DIRS")
    assert RunPaths.SUB_DIRS == ("figures", "checkpoints", "reports", "logs", "database", "exports")
    assert len(RunPaths.SUB_DIRS) == 6


# RUNPATHS: CREATION FACTORY
@pytest.mark.unit
def test_runpaths_create_basic(tmp_path):
    """Test RunPaths.create() with minimal valid arguments."""
    training_cfg = {"batch_size": 32, "learning_rate": 0.001, "epochs": 10}

    run_paths = RunPaths.create(
        dataset_slug="organcmnist",
        architecture_name="EfficientNet-B0",
        training_cfg=training_cfg,
        base_dir=tmp_path,
    )

    assert isinstance(run_paths, RunPaths)
    assert run_paths.dataset_slug == "organcmnist"
    assert run_paths.architecture_slug == "efficientnetb0"
    assert run_paths.root.parent == tmp_path


@pytest.mark.unit
def test_runpaths_create_uses_default_base_dir():
    """Test RunPaths.create() uses OUTPUTS_ROOT when base_dir not provided."""
    training_cfg = {"batch_size": 32}

    run_paths = RunPaths.create(
        dataset_slug="test",
        architecture_name="resnet",
        training_cfg=training_cfg,
    )

    assert run_paths.root.parent == OUTPUTS_ROOT


@pytest.mark.unit
def test_runpaths_create_normalizes_dataset_slug(tmp_path):
    """Test dataset_slug is normalized to lowercase."""
    training_cfg = {"batch_size": 32}

    run_paths = RunPaths.create(
        dataset_slug="OrganCMNIST",
        architecture_name="resnet",
        training_cfg=training_cfg,
        base_dir=tmp_path,
    )

    assert run_paths.dataset_slug == "organcmnist"


@pytest.mark.unit
def test_runpaths_create_normalizes_model_name(tmp_path):
    """Test model_name is sanitized (alphanumeric only, lowercase)."""
    training_cfg = {"batch_size": 32}

    test_cases = [
        ("EfficientNet-B0", "efficientnetb0"),
        ("ResNet_50", "resnet50"),
        ("VGG-16", "vgg16"),
        ("DenseNet-121", "densenet121"),
    ]

    for architecture_name, expected_slug in test_cases:
        run_paths = RunPaths.create(
            dataset_slug="test",
            architecture_name=architecture_name,
            training_cfg=training_cfg,
            base_dir=tmp_path,
        )
        assert run_paths.architecture_slug == expected_slug


@pytest.mark.unit
def test_runpaths_create_invalid_dataset_type():
    """Test RunPaths.create() raises ValueError for non-string dataset_slug."""
    training_cfg = {"batch_size": 32}

    with pytest.raises(ValueError, match="Expected string for dataset_slug"):
        RunPaths.create(
            dataset_slug=123,
            architecture_name="resnet",
            training_cfg=training_cfg,
        )


@pytest.mark.unit
def test_runpaths_create_invalid_model_type():
    """Test RunPaths.create() raises ValueError for non-string architecture_name."""
    training_cfg = {"batch_size": 32}

    with pytest.raises(ValueError, match="Expected string for architecture_name"):
        RunPaths.create(
            dataset_slug="test",
            architecture_name=123,
            training_cfg=training_cfg,
        )


# RUNPATHS: UNIQUE ID GENERATION
@pytest.mark.unit
def test_generate_unique_id_format():
    """Test _generate_unique_id() produces correct format: YYYYMMDD_dataset_model_hash."""
    ds_slug = "organcmnist"
    a_slug = "efficientnetb0"
    cfg = {"batch_size": 32, "learning_rate": 0.001}

    run_id = RunPaths._generate_unique_id(ds_slug, a_slug, cfg)

    parts = run_id.split("_")
    assert len(parts) == 4
    assert len(parts[0]) == 8
    assert parts[1] == ds_slug
    assert parts[2] == a_slug
    assert len(parts[3]) == 6


@pytest.mark.unit
def test_generate_unique_id_deterministic():
    """Test _generate_unique_id() produces same hash for identical configs + timestamp."""
    ds_slug = "test"
    a_slug = "architecture"
    fixed_ts = 1707400000.0  # Fixed timestamp for determinism
    cfg = {"batch_size": 32, "learning_rate": 0.001, "epochs": 10, "run_timestamp": fixed_ts}

    id1 = RunPaths._generate_unique_id(ds_slug, a_slug, cfg)
    id2 = RunPaths._generate_unique_id(ds_slug, a_slug, cfg)

    assert id1 == id2


@pytest.mark.unit
def test_generate_unique_id_different_configs():
    """Test _generate_unique_id() produces different hashes for different configs."""
    ds_slug = "test"
    a_slug = "architecture"
    cfg1 = {"batch_size": 32, "learning_rate": 0.001}
    cfg2 = {"batch_size": 64, "learning_rate": 0.001}

    id1 = RunPaths._generate_unique_id(ds_slug, a_slug, cfg1)
    id2 = RunPaths._generate_unique_id(ds_slug, a_slug, cfg2)

    assert id1 != id2


@pytest.mark.unit
def test_generate_unique_id_filters_non_hashable():
    """Test _generate_unique_id() filters out non-hashable types."""
    ds_slug = "test"
    a_slug = "architecture"
    cfg = {
        "batch_size": 32,
        "learning_rate": 0.001,
        "optimizer": {"type": "adam"},
        "callbacks": [{"early_stop": True}],
    }

    run_id = RunPaths._generate_unique_id(ds_slug, a_slug, cfg)
    assert isinstance(run_id, str)


@pytest.mark.unit
def test_generate_unique_id_empty_config():
    """Test _generate_unique_id() handles empty config dict."""
    ds_slug = "test"
    a_slug = "architecture"
    cfg = {}

    run_id = RunPaths._generate_unique_id(ds_slug, a_slug, cfg)
    assert isinstance(run_id, str)
    assert ds_slug in run_id
    assert a_slug in run_id


@pytest.mark.unit
def test_generate_unique_id_uses_blake2b():
    """Test _generate_unique_id() uses blake2b with digest_size=3."""
    ds_slug = "test"
    a_slug = "architecture"
    fixed_ts = 1707400000.0
    cfg = {"key": "value", "run_timestamp": fixed_ts}

    # Replicate the internal hashing logic
    hashable = {k: v for k, v in cfg.items() if isinstance(v, (int, float, str, bool, list))}
    hashable["_run_ts"] = fixed_ts
    params_json = json.dumps(hashable, sort_keys=True)
    expected_hash = hashlib.blake2b(params_json.encode(), digest_size=3).hexdigest()

    run_id = RunPaths._generate_unique_id(ds_slug, a_slug, cfg)

    assert expected_hash in run_id


# RUNPATHS: UNIQUENESS VIA TIMESTAMP
@pytest.mark.unit
def test_runpaths_unique_via_timestamp(tmp_path):
    """Test that different timestamps produce unique run IDs."""
    cfg1 = {"batch_size": 32, "run_timestamp": 1707400000.0}
    cfg2 = {"batch_size": 32, "run_timestamp": 1707400001.0}

    run1 = RunPaths.create(
        dataset_slug="test",
        architecture_name="architecture",
        training_cfg=cfg1,
        base_dir=tmp_path,
    )

    run2 = RunPaths.create(
        dataset_slug="test",
        architecture_name="architecture",
        training_cfg=cfg2,
        base_dir=tmp_path,
    )

    # Same config, different timestamps = different run IDs
    assert run1.run_id != run2.run_id


# RUNPATHS: DIRECTORY STRUCTURE
@pytest.mark.unit
def test_runpaths_creates_all_subdirectories(tmp_path):
    """Test RunPaths.create() physically creates all subdirectories."""
    training_cfg = {"batch_size": 32}

    run_paths = RunPaths.create(
        dataset_slug="test",
        architecture_name="architecture",
        training_cfg=training_cfg,
        base_dir=tmp_path,
    )

    assert run_paths.root.exists()
    assert run_paths.root.is_dir()

    for subdir_name in RunPaths.SUB_DIRS:
        subdir_path = run_paths.root / subdir_name
        assert subdir_path.exists(), f"Missing subdirectory: {subdir_name}"
        assert subdir_path.is_dir()


@pytest.mark.unit
def test_runpaths_path_attributes(tmp_path):
    """Test all path attributes are correctly set."""
    training_cfg = {"batch_size": 32}

    run_paths = RunPaths.create(
        dataset_slug="test",
        architecture_name="architecture",
        training_cfg=training_cfg,
        base_dir=tmp_path,
    )

    assert isinstance(run_paths.root, Path)
    assert isinstance(run_paths.figures, Path)
    assert isinstance(run_paths.checkpoints, Path)
    assert isinstance(run_paths.reports, Path)
    assert isinstance(run_paths.logs, Path)
    assert isinstance(run_paths.database, Path)
    assert run_paths.figures == run_paths.root / "figures"
    assert run_paths.checkpoints == run_paths.root / "checkpoints"
    assert run_paths.reports == run_paths.root / "reports"
    assert run_paths.logs == run_paths.root / "logs"
    assert run_paths.database == run_paths.root / "database"


# RUNPATHS: DYNAMIC PROPERTIES
@pytest.mark.unit
def test_best_model_path_property(tmp_path):
    """Test best_model_path property returns correct path."""
    training_cfg = {"batch_size": 32}

    run_paths = RunPaths.create(
        dataset_slug="test",
        architecture_name="ResNet-50",
        training_cfg=training_cfg,
        base_dir=tmp_path,
    )

    expected_path = run_paths.checkpoints / "best_resnet50.pth"
    assert run_paths.best_model_path == expected_path


@pytest.mark.unit
def test_final_report_path_property(tmp_path):
    """Test final_report_path property returns correct path."""
    training_cfg = {"batch_size": 32}

    run_paths = RunPaths.create(
        dataset_slug="test",
        architecture_name="architecture",
        training_cfg=training_cfg,
        base_dir=tmp_path,
    )

    expected_path = run_paths.reports / "training_summary.xlsx"
    assert run_paths.final_report_path == expected_path


@pytest.mark.unit
def test_get_fig_path_method(tmp_path):
    """Test get_fig_path() method returns correct figure path."""
    training_cfg = {"batch_size": 32}

    run_paths = RunPaths.create(
        dataset_slug="test",
        architecture_name="architecture",
        training_cfg=training_cfg,
        base_dir=tmp_path,
    )

    fig_path = run_paths.get_fig_path("confusion_matrix.png")
    assert fig_path == run_paths.figures / "confusion_matrix.png"

    fig_path2 = run_paths.get_fig_path("roc_curve.pdf")
    assert fig_path2 == run_paths.figures / "roc_curve.pdf"


@pytest.mark.unit
def test_get_config_path_method(tmp_path):
    """Test get_config_path() method returns correct config path."""
    training_cfg = {"batch_size": 32}

    run_paths = RunPaths.create(
        dataset_slug="test",
        architecture_name="architecture",
        training_cfg=training_cfg,
        base_dir=tmp_path,
    )

    config_path = run_paths.get_config_path()
    assert config_path == run_paths.reports / "config.yaml"


@pytest.mark.unit
def test_get_db_path_method(tmp_path):
    """Test get_db_path() method returns correct database path."""
    training_cfg = {"batch_size": 32}

    run_paths = RunPaths.create(
        dataset_slug="test",
        architecture_name="architecture",
        training_cfg=training_cfg,
        base_dir=tmp_path,
    )

    db_path = run_paths.get_db_path()
    assert db_path == run_paths.database / "study.db"


# RUNPATHS: IMMUTABILITY
@pytest.mark.unit
def test_runpaths_is_frozen(tmp_path):
    """Test RunPaths instances are immutable after creation."""
    training_cfg = {"batch_size": 32}

    run_paths = RunPaths.create(
        dataset_slug="test",
        architecture_name="architecture",
        training_cfg=training_cfg,
        base_dir=tmp_path,
    )

    with pytest.raises(ValidationError):
        run_paths.run_id = "new_id"

    with pytest.raises(ValidationError):
        run_paths.dataset_slug = "new_dataset"


# RUNPATHS: STRING REPRESENTATION
@pytest.mark.unit
def test_runpaths_repr(tmp_path):
    """Test __repr__ provides useful debug information."""
    training_cfg = {"batch_size": 32}

    run_paths = RunPaths.create(
        dataset_slug="organcmnist",
        architecture_name="ResNet_18_Adapted",
        training_cfg=training_cfg,
        base_dir=tmp_path,
    )

    repr_str = repr(run_paths)

    assert "RunPaths" in repr_str
    assert run_paths.run_id in repr_str
    assert "root=" in repr_str


# RUNPATHS: EDGE CASES
@pytest.mark.unit
def test_runpaths_create_with_special_characters_in_model(tmp_path):
    """Test model_name with various special characters is properly sanitized."""
    training_cfg = {"batch_size": 32}

    special_names = [
        ("Model@2024", "model2024"),
        ("Net#123", "net123"),
        ("Arch$v2", "archv2"),
        ("Test%Model", "testmodel"),
    ]

    for architecture_name, expected_slug in special_names:
        run_paths = RunPaths.create(
            dataset_slug="test",
            architecture_name=architecture_name,
            training_cfg=training_cfg,
            base_dir=tmp_path,
        )
        assert run_paths.architecture_slug == expected_slug


@pytest.mark.unit
def test_runpaths_create_with_empty_model_name(tmp_path):
    """Test empty model_name after sanitization."""
    training_cfg = {"batch_size": 32}

    run_paths = RunPaths.create(
        dataset_slug="test",
        architecture_name="@#$%",
        training_cfg=training_cfg,
        base_dir=tmp_path,
    )

    assert run_paths.architecture_slug == ""
    assert "test" in run_paths.run_id


@pytest.mark.unit
def test_runpaths_create_with_complex_training_config(tmp_path):
    """Test RunPaths.create() with complex nested training config."""
    training_cfg = {
        "batch_size": 32,
        "learning_rate": 0.001,
        "optimizer": "adam",
        "epochs": 100,
        "use_augmentation": True,
        "dropout_rate": 0.5,
        "weight_decay": 1e-5,
    }

    run_paths = RunPaths.create(
        dataset_slug="test",
        architecture_name="architecture",
        training_cfg=training_cfg,
        base_dir=tmp_path,
    )

    assert isinstance(run_paths, RunPaths)
    assert run_paths.root.exists()


# RUNPATHS: INTEGRATION TESTS
@pytest.mark.integration
def test_runpaths_full_workflow(tmp_path):
    """Test complete RunPaths workflow from creation to artifact saving."""
    training_cfg = {
        "batch_size": 32,
        "learning_rate": 0.001,
        "epochs": 10,
    }

    run_paths = RunPaths.create(
        dataset_slug="organcmnist",
        architecture_name="EfficientNet_B0",
        training_cfg=training_cfg,
        base_dir=tmp_path,
    )

    assert run_paths.root.exists()

    (run_paths.checkpoints / "checkpoint.pth").touch()
    (run_paths.reports / "metrics.json").touch()
    (run_paths.figures / "plot.png").touch()

    assert (run_paths.checkpoints / "checkpoint.pth").exists()
    assert (run_paths.reports / "metrics.json").exists()
    assert (run_paths.figures / "plot.png").exists()


@pytest.mark.integration
def test_multiple_runs_different_configs(tmp_path):
    """Test creating multiple runs with different configs produces unique directories."""
    configs = [
        {"batch_size": 32, "learning_rate": 0.001},
        {"batch_size": 64, "learning_rate": 0.001},
        {"batch_size": 32, "learning_rate": 0.01},
    ]

    run_ids = []
    for cfg in configs:
        run_paths = RunPaths.create(
            dataset_slug="test",
            architecture_name="architecture",
            training_cfg=cfg,
            base_dir=tmp_path,
        )
        run_ids.append(run_paths.run_id)

    assert len(run_ids) == len(set(run_ids))

    for run_id in run_ids:
        assert (tmp_path / run_id).exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
