"""
Test Suite for OptunaConfig.

Tests Optuna study configuration, early stopping parameters,
sampler/pruner selection, and storage backend configuration.
"""

# Standard Imports
import warnings
from pathlib import Path

# Third-Party Imports
import pytest
from pydantic import ValidationError

# Internal Imports
from orchard.core.config import OptunaConfig


# OPTUNA CONFIG: DEFAULTS
@pytest.mark.unit
def test_optuna_config_defaults():
    """Test OptunaConfig with default values."""
    config = OptunaConfig()

    assert config.study_name == "vision_optimization"
    assert config.n_trials == 50
    assert config.epochs == 15
    assert config.metric_name == "auc"
    assert config.direction == "maximize"
    assert config.sampler_type == "tpe"
    assert config.enable_early_stopping is True
    assert config.enable_pruning is True


@pytest.mark.unit
def test_optuna_config_custom_values():
    """Test OptunaConfig with custom parameters."""
    config = OptunaConfig(
        study_name="custom_study",
        n_trials=100,
        epochs=30,
        metric_name="accuracy",
        direction="maximize",
    )

    assert config.study_name == "custom_study"
    assert config.n_trials == 100
    assert config.epochs == 30
    assert config.metric_name == "accuracy"


# OPTUNA CONFIG: VALIDATION
@pytest.mark.unit
def test_invalid_metric_name_rejected():
    """Test invalid metric_name is rejected."""
    with pytest.raises(ValidationError, match="metric_name.*invalid"):
        OptunaConfig(metric_name="invalid_metric")


@pytest.mark.unit
def test_valid_metric_names():
    """Test valid metric names are accepted."""
    for metric in ["auc", "accuracy", "loss"]:
        config = OptunaConfig(metric_name=metric)
        assert config.metric_name == metric


@pytest.mark.unit
def test_pruning_warmup_exceeds_epochs_rejected():
    """Test pruning_warmup_epochs >= epochs is rejected."""
    with pytest.raises(ValidationError, match="pruning_warmup"):
        OptunaConfig(epochs=10, pruning_warmup_epochs=10)


@pytest.mark.unit
def test_pruning_warmup_valid():
    """Test valid pruning_warmup_epochs configuration."""
    config = OptunaConfig(epochs=20, pruning_warmup_epochs=5)
    assert config.pruning_warmup_epochs == 5


@pytest.mark.unit
def test_n_trials_positive():
    """Test n_trials must be positive."""
    config = OptunaConfig(n_trials=1)
    assert config.n_trials == 1

    with pytest.raises(ValidationError):
        OptunaConfig(n_trials=0)

    with pytest.raises(ValidationError):
        OptunaConfig(n_trials=-5)


@pytest.mark.unit
def test_epochs_positive():
    """Test epochs must be positive and > warmup."""
    config = OptunaConfig(epochs=10, pruning_warmup_epochs=2)
    assert config.epochs == 10

    with pytest.raises(ValidationError):
        OptunaConfig(epochs=3, pruning_warmup_epochs=5)


@pytest.mark.unit
def test_show_progress_bar_warning():
    """Test that a warning is issued if show_progress_bar=True and n_jobs>1."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        OptunaConfig(show_progress_bar=True, n_jobs=2)
        assert any("may corrupt tqdm output" in str(warning.message) for warning in w)


# OPTUNA CONFIG: EARLY STOPPING
@pytest.mark.unit
def test_early_stopping_configuration():
    """Test early stopping parameter configuration."""
    config = OptunaConfig(
        enable_early_stopping=True, early_stopping_threshold=0.999, early_stopping_patience=2
    )

    assert config.enable_early_stopping is True
    assert config.early_stopping_threshold == 0.999
    assert config.early_stopping_patience == 2


@pytest.mark.unit
def test_early_stopping_threshold_bounds():
    """Test early_stopping_threshold accepts values >= 0."""
    # Valid
    config = OptunaConfig(early_stopping_threshold=0.0)
    assert config.early_stopping_threshold == 0.0

    config = OptunaConfig(early_stopping_threshold=0.999)
    assert config.early_stopping_threshold == 0.999

    config = OptunaConfig(early_stopping_threshold=1.5)
    assert config.early_stopping_threshold == 1.5


@pytest.mark.unit
def test_early_stopping_patience_positive():
    """Test early_stopping_patience must be positive."""
    config = OptunaConfig(early_stopping_patience=1)
    assert config.early_stopping_patience == 1

    with pytest.raises(ValidationError):
        OptunaConfig(early_stopping_patience=0)


# OPTUNA CONFIG: SAMPLER AND PRUNER
@pytest.mark.unit
def test_sampler_types():
    """Test valid sampler types are accepted."""
    for sampler in ["tpe", "cmaes", "random", "grid"]:
        config = OptunaConfig(sampler_type=sampler)
        assert config.sampler_type == sampler


@pytest.mark.unit
def test_pruner_types():
    """Test valid pruner types are accepted."""
    for pruner in ["median", "percentile", "hyperband", "none"]:
        config = OptunaConfig(pruner_type=pruner)
        assert config.pruner_type == pruner


@pytest.mark.unit
def test_pruning_can_be_disabled():
    """Test pruning can be disabled."""
    config = OptunaConfig(enable_pruning=False)

    assert config.enable_pruning is False


# OPTUNA CONFIG: STORAGE BACKEND
@pytest.mark.unit
def test_storage_type_options():
    """Test valid storage types are accepted."""
    for storage in ["memory", "sqlite", "postgresql"]:
        if storage == "postgresql":
            config = OptunaConfig(storage_type=storage, storage_path="postgres://localhost")
        else:
            config = OptunaConfig(storage_type=storage)
        assert config.storage_type == storage


@pytest.mark.unit
def test_postgresql_without_storage_path_rejected():
    """Test PostgreSQL storage requires storage_path."""
    with pytest.raises(ValidationError, match="PostgreSQL.*storage_path"):
        OptunaConfig(storage_type="postgresql", storage_path=None)


@pytest.mark.unit
def test_postgresql_with_storage_path_valid():
    """Test PostgreSQL with storage_path is valid."""
    config = OptunaConfig(
        storage_type="postgresql", storage_path="postgresql://user:pass@localhost/db"
    )

    assert config.storage_type == "postgresql"
    assert config.storage_path is not None


@pytest.mark.unit
def test_get_storage_url_memory():
    """Test get_storage_url() for memory backend."""
    config = OptunaConfig(storage_type="memory")

    class MockPaths:
        def get_db_path(self):
            return Path("/tmp/study.db")

    url = config.get_storage_url(MockPaths())
    assert url is None


@pytest.mark.unit
def test_get_storage_url_sqlite():
    """Test get_storage_url() for SQLite backend."""
    config = OptunaConfig(storage_type="sqlite")

    class MockPaths:
        def get_db_path(self):
            return Path("/tmp/test_study.db")

    url = config.get_storage_url(MockPaths())
    assert url.startswith("sqlite:///")
    assert "test_study.db" in url


@pytest.mark.unit
def test_get_storage_url_postgresql():
    """Test get_storage_url() for PostgreSQL backend."""
    config = OptunaConfig(storage_type="postgresql", storage_path="postgresql://localhost/optuna")

    class MockPaths:
        pass

    url = config.get_storage_url(MockPaths())

    assert "localhost" in str(url)
    assert "optuna" in str(url)


# OPTUNA CONFIG: DIRECTION
@pytest.mark.unit
def test_direction_maximize():
    """Test direction='maximize' is accepted."""
    config = OptunaConfig(direction="maximize")
    assert config.direction == "maximize"


@pytest.mark.unit
def test_direction_minimize():
    """Test direction='minimize' is accepted."""
    config = OptunaConfig(direction="minimize")
    assert config.direction == "minimize"


# OPTUNA CONFIG: FROM ARGS
@pytest.mark.unit
def test_from_args(optuna_args):
    """Test OptunaConfig.from_args() factory."""
    config = OptunaConfig.from_args(optuna_args)

    assert config.study_name == "test_study"
    assert config.n_trials == 10
    assert config.epochs == 15
    assert config.metric_name == "auc"


@pytest.mark.unit
def test_from_args_partial():
    """Test from_args() with partial arguments uses defaults."""
    from argparse import Namespace

    args = Namespace(n_trials=20)
    config = OptunaConfig.from_args(args)

    assert config.n_trials == 20
    assert config.study_name == "vision_optimization"  # Default


# OPTUNA CONFIG: IMMUTABILITY
@pytest.mark.unit
def test_config_is_frozen():
    """Test OptunaConfig is immutable after creation."""
    config = OptunaConfig()

    with pytest.raises(ValidationError):
        config.n_trials = 100


@pytest.mark.unit
def test_config_forbids_extra_fields():
    """Test OptunaConfig rejects unknown fields."""
    with pytest.raises(ValidationError):
        OptunaConfig(unknown_param="value")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
