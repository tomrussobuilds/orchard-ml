"""
Test Suite for RunPaths Dynamic Directory Management.

Tests atomic run isolation, unique ID generation, directory creation,
and path resolution for experiment artifacts.
"""

# Standard Imports
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

# Third-Party Imports
import pytest
from pydantic import ValidationError

# Internal Imports
from orchard.core.paths import OUTPUTS_ROOT, RunPaths


# RUNPATHS: CLASS CONSTANTS
@pytest.mark.unit
def test_sub_dirs_constant():
    """Test SUB_DIRS class constant contains all required subdirectories."""
    assert hasattr(RunPaths, "SUB_DIRS")
    assert RunPaths.SUB_DIRS == ("figures", "models", "reports", "logs", "database")
    assert len(RunPaths.SUB_DIRS) == 5


# RUNPATHS: CREATION FACTORY
@pytest.mark.unit
def test_runpaths_create_basic(tmp_path):
    """Test RunPaths.create() with minimal valid arguments."""
    training_cfg = {"batch_size": 32, "learning_rate": 0.001, "epochs": 10}

    run_paths = RunPaths.create(
        dataset_slug="organcmnist",
        model_name="EfficientNet-B0",
        training_cfg=training_cfg,
        base_dir=tmp_path,
    )

    assert isinstance(run_paths, RunPaths)
    assert run_paths.dataset_slug == "organcmnist"
    assert run_paths.model_slug == "efficientnetb0"
    assert run_paths.root.parent == tmp_path


@pytest.mark.unit
def test_runpaths_create_uses_default_base_dir():
    """Test RunPaths.create() uses OUTPUTS_ROOT when base_dir not provided."""
    training_cfg = {"batch_size": 32}

    run_paths = RunPaths.create(
        dataset_slug="test",
        model_name="resnet",
        training_cfg=training_cfg,
    )

    assert run_paths.root.parent == OUTPUTS_ROOT


@pytest.mark.unit
def test_runpaths_create_normalizes_dataset_slug():
    """Test dataset_slug is normalized to lowercase."""
    training_cfg = {"batch_size": 32}

    run_paths = RunPaths.create(
        dataset_slug="OrganCMNIST",
        model_name="resnet",
        training_cfg=training_cfg,
        base_dir=Path("/tmp"),
    )

    assert run_paths.dataset_slug == "organcmnist"


@pytest.mark.unit
def test_runpaths_create_normalizes_model_name():
    """Test model_name is sanitized (alphanumeric only, lowercase)."""
    training_cfg = {"batch_size": 32}

    test_cases = [
        ("EfficientNet-B0", "efficientnetb0"),
        ("ResNet_50", "resnet50"),
        ("VGG-16", "vgg16"),
        ("DenseNet-121", "densenet121"),
    ]

    for model_name, expected_slug in test_cases:
        run_paths = RunPaths.create(
            dataset_slug="test",
            model_name=model_name,
            training_cfg=training_cfg,
            base_dir=Path("/tmp"),
        )
        assert run_paths.model_slug == expected_slug


@pytest.mark.unit
def test_runpaths_create_invalid_dataset_type():
    """Test RunPaths.create() raises ValueError for non-string dataset_slug."""
    training_cfg = {"batch_size": 32}

    with pytest.raises(ValueError, match="Expected string for dataset_slug"):
        RunPaths.create(
            dataset_slug=123,  # Invalid type
            model_name="resnet",
            training_cfg=training_cfg,
        )


@pytest.mark.unit
def test_runpaths_create_invalid_model_type():
    """Test RunPaths.create() raises ValueError for non-string model_name."""
    training_cfg = {"batch_size": 32}

    with pytest.raises(ValueError, match="Expected string for model_name"):
        RunPaths.create(
            dataset_slug="test",
            model_name=123,  # Invalid type
            training_cfg=training_cfg,
        )


# RUNPATHS: UNIQUE ID GENERATION
@pytest.mark.unit
def test_generate_unique_id_format():
    """Test _generate_unique_id() produces correct format: YYYYMMDD_dataset_model_hash."""
    ds_slug = "organcmnist"
    m_slug = "efficientnetb0"
    cfg = {"batch_size": 32, "learning_rate": 0.001}

    run_id = RunPaths._generate_unique_id(ds_slug, m_slug, cfg)

    # Should match: 20260119_organcmnist_efficientnetb0_a3f7c2
    parts = run_id.split("_")
    assert len(parts) == 4
    assert len(parts[0]) == 8  # YYYYMMDD
    assert parts[1] == ds_slug
    assert parts[2] == m_slug
    assert len(parts[3]) == 6  # 6-char hash


@pytest.mark.unit
def test_generate_unique_id_deterministic():
    """Test _generate_unique_id() produces same hash for identical configs."""
    ds_slug = "test"
    m_slug = "model"
    cfg = {"batch_size": 32, "learning_rate": 0.001, "epochs": 10}

    id1 = RunPaths._generate_unique_id(ds_slug, m_slug, cfg)
    id2 = RunPaths._generate_unique_id(ds_slug, m_slug, cfg)

    assert id1 == id2


@pytest.mark.unit
def test_generate_unique_id_different_configs():
    """Test _generate_unique_id() produces different hashes for different configs."""
    ds_slug = "test"
    m_slug = "model"
    cfg1 = {"batch_size": 32, "learning_rate": 0.001}
    cfg2 = {"batch_size": 64, "learning_rate": 0.001}

    id1 = RunPaths._generate_unique_id(ds_slug, m_slug, cfg1)
    id2 = RunPaths._generate_unique_id(ds_slug, m_slug, cfg2)

    assert id1 != id2


@pytest.mark.unit
def test_generate_unique_id_filters_non_hashable():
    """Test _generate_unique_id() filters out non-hashable types."""
    ds_slug = "test"
    m_slug = "model"
    cfg = {
        "batch_size": 32,  # Hashable
        "learning_rate": 0.001,  # Hashable
        "optimizer": {"type": "adam"},  # Non-hashable dict
        "callbacks": [{"early_stop": True}],  # List of dicts (non-hashable)
    }

    # Should not raise error, only hashes primitives
    run_id = RunPaths._generate_unique_id(ds_slug, m_slug, cfg)
    assert isinstance(run_id, str)


@pytest.mark.unit
def test_generate_unique_id_empty_config():
    """Test _generate_unique_id() handles empty config dict."""
    ds_slug = "test"
    m_slug = "model"
    cfg = {}

    run_id = RunPaths._generate_unique_id(ds_slug, m_slug, cfg)
    assert isinstance(run_id, str)
    assert ds_slug in run_id
    assert m_slug in run_id


@pytest.mark.unit
def test_generate_unique_id_uses_blake2b():
    """Test _generate_unique_id() uses blake2b with digest_size=3."""
    ds_slug = "test"
    m_slug = "model"
    cfg = {"key": "value"}

    # Manually compute expected hash
    hashable = {k: v for k, v in cfg.items() if isinstance(v, (int, float, str, bool, list))}
    params_json = json.dumps(hashable, sort_keys=True)
    expected_hash = hashlib.blake2b(params_json.encode(), digest_size=3).hexdigest()

    run_id = RunPaths._generate_unique_id(ds_slug, m_slug, cfg)

    assert expected_hash in run_id


# RUNPATHS: COLLISION HANDLING
@pytest.mark.unit
def test_runpaths_create_handles_collision(tmp_path):
    """Test RunPaths.create() appends timestamp if directory already exists."""
    training_cfg = {"batch_size": 32}

    # Create first run
    run1 = RunPaths.create(
        dataset_slug="test",
        model_name="model",
        training_cfg=training_cfg,
        base_dir=tmp_path,
    )

    # Manually create the directory to simulate collision
    run1.root.mkdir(parents=True, exist_ok=True)

    # Create second run with same config (should append timestamp)
    with patch("time.strftime", return_value="123456"):
        run2 = RunPaths.create(
            dataset_slug="test",
            model_name="model",
            training_cfg=training_cfg,
            base_dir=tmp_path,
        )

    assert run1.run_id != run2.run_id
    assert run2.run_id.startswith("123456")


# RUNPATHS: DIRECTORY STRUCTURE
@pytest.mark.unit
def test_runpaths_creates_all_subdirectories(tmp_path):
    """Test RunPaths.create() physically creates all subdirectories."""
    training_cfg = {"batch_size": 32}

    run_paths = RunPaths.create(
        dataset_slug="test",
        model_name="model",
        training_cfg=training_cfg,
        base_dir=tmp_path,
    )

    # Verify root exists
    assert run_paths.root.exists()
    assert run_paths.root.is_dir()

    # Verify all subdirectories exist
    for subdir_name in RunPaths.SUB_DIRS:
        subdir_path = run_paths.root / subdir_name
        assert subdir_path.exists(), f"Missing subdirectory: {subdir_name}"
        assert subdir_path.is_dir()


@pytest.mark.unit
def test_runpaths_path_attributes():
    """Test all path attributes are correctly set."""
    training_cfg = {"batch_size": 32}

    run_paths = RunPaths.create(
        dataset_slug="test",
        model_name="model",
        training_cfg=training_cfg,
        base_dir=Path("/tmp"),
    )

    # Check all Path attributes
    assert isinstance(run_paths.root, Path)
    assert isinstance(run_paths.figures, Path)
    assert isinstance(run_paths.models, Path)
    assert isinstance(run_paths.reports, Path)
    assert isinstance(run_paths.logs, Path)
    assert isinstance(run_paths.database, Path)

    # Check path relationships
    assert run_paths.figures == run_paths.root / "figures"
    assert run_paths.models == run_paths.root / "models"
    assert run_paths.reports == run_paths.root / "reports"
    assert run_paths.logs == run_paths.root / "logs"
    assert run_paths.database == run_paths.root / "database"


# RUNPATHS: DYNAMIC PROPERTIES
@pytest.mark.unit
def test_best_model_path_property():
    """Test best_model_path property returns correct path."""
    training_cfg = {"batch_size": 32}

    run_paths = RunPaths.create(
        dataset_slug="test",
        model_name="ResNet-50",
        training_cfg=training_cfg,
        base_dir=Path("/tmp"),
    )

    expected_path = run_paths.models / "best_resnet50.pth"
    assert run_paths.best_model_path == expected_path


@pytest.mark.unit
def test_final_report_path_property():
    """Test final_report_path property returns correct path."""
    training_cfg = {"batch_size": 32}

    run_paths = RunPaths.create(
        dataset_slug="test",
        model_name="model",
        training_cfg=training_cfg,
        base_dir=Path("/tmp"),
    )

    expected_path = run_paths.reports / "training_summary.xlsx"
    assert run_paths.final_report_path == expected_path


@pytest.mark.unit
def test_get_fig_path_method():
    """Test get_fig_path() method returns correct figure path."""
    training_cfg = {"batch_size": 32}

    run_paths = RunPaths.create(
        dataset_slug="test",
        model_name="model",
        training_cfg=training_cfg,
        base_dir=Path("/tmp"),
    )

    fig_path = run_paths.get_fig_path("confusion_matrix.png")
    assert fig_path == run_paths.figures / "confusion_matrix.png"

    fig_path2 = run_paths.get_fig_path("roc_curve.pdf")
    assert fig_path2 == run_paths.figures / "roc_curve.pdf"


@pytest.mark.unit
def test_get_config_path_method():
    """Test get_config_path() method returns correct config path."""
    training_cfg = {"batch_size": 32}

    run_paths = RunPaths.create(
        dataset_slug="test",
        model_name="model",
        training_cfg=training_cfg,
        base_dir=Path("/tmp"),
    )

    config_path = run_paths.get_config_path()
    assert config_path == run_paths.reports / "config.yaml"


@pytest.mark.unit
def test_get_db_path_method():
    """Test get_db_path() method returns correct database path."""
    training_cfg = {"batch_size": 32}

    run_paths = RunPaths.create(
        dataset_slug="test",
        model_name="model",
        training_cfg=training_cfg,
        base_dir=Path("/tmp"),
    )

    db_path = run_paths.get_db_path()
    assert db_path == run_paths.database / "study.db"


# RUNPATHS: IMMUTABILITY
@pytest.mark.unit
def test_runpaths_is_frozen():
    """Test RunPaths instances are immutable after creation."""
    training_cfg = {"batch_size": 32}

    run_paths = RunPaths.create(
        dataset_slug="test",
        model_name="model",
        training_cfg=training_cfg,
        base_dir=Path("/tmp"),
    )

    # Attempt to modify should raise ValidationError
    with pytest.raises(ValidationError):
        run_paths.run_id = "new_id"

    with pytest.raises(ValidationError):
        run_paths.dataset_slug = "new_dataset"


# RUNPATHS: STRING REPRESENTATION
@pytest.mark.unit
def test_runpaths_repr():
    """Test __repr__ provides useful debug information."""
    training_cfg = {"batch_size": 32}

    run_paths = RunPaths.create(
        dataset_slug="organcmnist",
        model_name="ResNet",
        training_cfg=training_cfg,
        base_dir=Path("/tmp"),
    )

    repr_str = repr(run_paths)

    assert "RunPaths" in repr_str
    assert run_paths.run_id in repr_str
    assert "root=" in repr_str


# RUNPATHS: EDGE CASES
@pytest.mark.unit
def test_runpaths_create_with_special_characters_in_model():
    """Test model_name with various special characters is properly sanitized."""
    training_cfg = {"batch_size": 32}

    special_names = [
        ("Model@2024", "model2024"),
        ("Net#123", "net123"),
        ("Arch$v2", "archv2"),
        ("Test%Model", "testmodel"),
    ]

    for model_name, expected_slug in special_names:
        run_paths = RunPaths.create(
            dataset_slug="test",
            model_name=model_name,
            training_cfg=training_cfg,
            base_dir=Path("/tmp"),
        )
        assert run_paths.model_slug == expected_slug


@pytest.mark.unit
def test_runpaths_create_with_empty_model_name():
    """Test empty model_name after sanitization."""
    training_cfg = {"batch_size": 32}

    # Model name with only special characters becomes empty after sanitization
    run_paths = RunPaths.create(
        dataset_slug="test",
        model_name="@#$%",
        training_cfg=training_cfg,
        base_dir=Path("/tmp"),
    )

    assert run_paths.model_slug == ""
    assert "test" in run_paths.run_id


@pytest.mark.unit
def test_runpaths_create_with_complex_training_config():
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
        model_name="model",
        training_cfg=training_cfg,
        base_dir=Path("/tmp"),
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

    # Create run paths
    run_paths = RunPaths.create(
        dataset_slug="organcmnist",
        model_name="EfficientNet-B0",
        training_cfg=training_cfg,
        base_dir=tmp_path,
    )

    # Verify structure
    assert run_paths.root.exists()

    # Simulate saving artifacts
    (run_paths.models / "checkpoint.pth").touch()
    (run_paths.reports / "metrics.json").touch()
    (run_paths.figures / "plot.png").touch()

    # Verify artifacts exist
    assert (run_paths.models / "checkpoint.pth").exists()
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
            model_name="model",
            training_cfg=cfg,
            base_dir=tmp_path,
        )
        run_ids.append(run_paths.run_id)

    # All run IDs should be unique
    assert len(run_ids) == len(set(run_ids))

    # All directories should exist
    for run_id in run_ids:
        assert (tmp_path / run_id).exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
