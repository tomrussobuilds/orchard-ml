"""
Test Suite for OptunaConfig.

Tests Optuna study configuration, early stopping parameters,
sampler/pruner selection, and storage backend configuration.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
from pydantic import ValidationError

from orchard.core.config import OptunaConfig


# Do not perform equality checks with floating point values.
# OPTUNA CONFIG: DEFAULTS
@pytest.mark.unit
def test_optuna_config_defaults():
    """Test OptunaConfig with default values."""
    config = OptunaConfig()

    assert config.study_name == "vision_optimization"
    assert config.n_trials == 50
    assert config.epochs == 15
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
        direction="maximize",
    )

    assert config.study_name == "custom_study"
    assert config.n_trials == 100
    assert config.epochs == 30


# OPTUNA CONFIG: VALIDATION
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
    assert config.early_stopping_threshold == pytest.approx(0.999)
    assert config.early_stopping_patience == 2


@pytest.mark.unit
def test_early_stopping_threshold_bounds():
    """Test early_stopping_threshold accepts values >= 0."""
    # Valid
    config = OptunaConfig(early_stopping_threshold=0.0)
    assert config.early_stopping_threshold == pytest.approx(0.0)

    config = OptunaConfig(early_stopping_threshold=0.999)
    assert config.early_stopping_threshold == pytest.approx(0.999)

    config = OptunaConfig(early_stopping_threshold=1.5)
    assert config.early_stopping_threshold == pytest.approx(1.5)


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
    for sampler in ["tpe", "cmaes", "random"]:
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
            config = OptunaConfig(storage_type=storage, postgresql_url="postgresql://localhost/db")
        else:
            config = OptunaConfig(storage_type=storage)
        assert config.storage_type == storage


@pytest.mark.unit
def test_postgresql_without_url_rejected():
    """Test PostgreSQL storage requires postgresql_url."""
    with pytest.raises(ValidationError, match="PostgreSQL.*postgresql_url"):
        OptunaConfig(storage_type="postgresql")


@pytest.mark.unit
def test_postgresql_with_url_valid():
    """Test PostgreSQL with postgresql_url is valid."""
    config = OptunaConfig(
        storage_type="postgresql", postgresql_url="postgresql://user:pass@localhost/db"
    )

    assert config.storage_type == "postgresql"
    assert config.postgresql_url == "postgresql://user:pass@localhost/db"


@pytest.mark.unit
def test_postgresql_url_invalid_scheme_rejected():
    """Test postgresql_url must start with postgresql:// or postgresql+."""
    with pytest.raises(ValidationError, match="must start with"):
        OptunaConfig(storage_type="postgresql", postgresql_url="mysql://localhost/db")


@pytest.mark.unit
def test_postgresql_url_without_postgresql_storage_rejected():
    """Test postgresql_url cannot be set when storage_type is not postgresql."""
    with pytest.raises(ValidationError, match="postgresql_url is set but"):
        OptunaConfig(storage_type="sqlite", postgresql_url="postgresql://localhost/db")


@pytest.mark.unit
def test_get_storage_url_memory():
    """Test get_storage_url() for memory backend."""
    config = OptunaConfig(storage_type="memory")

    class MockPaths:
        def get_db_path(self):
            return Path("/mock/study.db")

    url = config.get_storage_url(MockPaths())
    assert url is None


@pytest.mark.unit
def test_get_storage_url_sqlite():
    """Test get_storage_url() for SQLite backend."""
    config = OptunaConfig(storage_type="sqlite")

    class MockPaths:
        def get_db_path(self):
            return Path("/mock/test_study.db")

    url = config.get_storage_url(MockPaths())
    assert url.startswith("sqlite:///")
    assert "test_study.db" in url


@pytest.mark.unit
def test_get_storage_url_postgresql():
    """Test get_storage_url() for PostgreSQL backend."""
    config = OptunaConfig(storage_type="postgresql", postgresql_url="postgresql://localhost/optuna")

    class MockPaths:
        pass

    url = config.get_storage_url(MockPaths())

    assert url == "postgresql://localhost/optuna"


@pytest.mark.unit
def test_get_storage_url_sqlite_with_custom_path(tmp_path):
    """Test get_storage_url() for SQLite with custom storage_path."""
    custom_db = tmp_path / "custom.db"
    config = OptunaConfig(storage_type="sqlite", storage_path=str(custom_db))

    class MockPaths:
        def get_db_path(self):
            return Path("/mock/default_study.db")

    url = config.get_storage_url(MockPaths())

    assert url.startswith("sqlite:///")
    assert "custom.db" in url
    assert "default_study.db" not in url


@pytest.mark.unit
def test_get_storage_url_unknown_storage_type():
    """Test get_storage_url() raises ValueError for unknown storage type."""
    config = OptunaConfig(storage_type="sqlite")

    object.__setattr__(config, "storage_type", "invalid_backend")

    class MockPaths:
        def get_db_path(self):
            return Path("/mock/study.db")

    with pytest.raises(ValueError, match="Unknown storage type"):
        config.get_storage_url(MockPaths())


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


# OPTUNA CONFIG: MODEL SEARCH
@pytest.mark.unit
def test_enable_model_search_default():
    """Test enable_model_search defaults to False."""
    config = OptunaConfig()
    assert config.enable_model_search is False


@pytest.mark.unit
def test_enable_model_search_can_be_enabled():
    """Test enable_model_search can be set to True."""
    config = OptunaConfig(enable_model_search=True)
    assert config.enable_model_search is True


# OPTUNA CONFIG: MODEL POOL
@pytest.mark.unit
def test_model_pool_defaults_to_none():
    """Test model_pool defaults to None."""
    config = OptunaConfig()
    assert config.model_pool is None


@pytest.mark.unit
def test_model_pool_valid_with_model_search():
    """Test model_pool is accepted when enable_model_search=True."""
    config = OptunaConfig(
        enable_model_search=True,
        model_pool=["resnet_18", "efficientnet_b0"],
    )
    assert config.model_pool == ["resnet_18", "efficientnet_b0"]


@pytest.mark.unit
def test_model_pool_without_model_search_rejected():
    """Test model_pool requires enable_model_search=True."""
    with pytest.raises(ValidationError, match="model_pool requires enable_model_search"):
        OptunaConfig(
            enable_model_search=False,
            model_pool=["resnet_18", "mini_cnn"],
        )


@pytest.mark.unit
def test_model_pool_single_entry_rejected():
    """Test model_pool must contain at least 2 architectures."""
    with pytest.raises(ValidationError, match="at least 2 architectures"):
        OptunaConfig(
            enable_model_search=True,
            model_pool=["resnet_18"],
        )


@pytest.mark.unit
def test_model_pool_empty_list_rejected():
    """Test model_pool rejects empty list."""
    with pytest.raises(ValidationError, match="at least 2 architectures"):
        OptunaConfig(
            enable_model_search=True,
            model_pool=[],
        )


@pytest.mark.unit
def test_model_pool_with_timm_prefix():
    """Test model_pool accepts timm/ prefixed names."""
    config = OptunaConfig(
        enable_model_search=True,
        model_pool=["resnet_18", "timm/mobilenetv3_small_100"],
    )
    assert "timm/mobilenetv3_small_100" in config.model_pool


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


# OPTUNA CONFIG: SEARCH SPACE OVERRIDES
@pytest.mark.unit
def test_search_space_overrides_default():
    """Test OptunaConfig includes default SearchSpaceOverrides."""
    config = OptunaConfig()

    assert config.search_space_overrides is not None
    assert config.search_space_overrides.learning_rate.low == pytest.approx(1e-5)
    assert config.search_space_overrides.learning_rate.high == pytest.approx(1e-2)
    assert config.search_space_overrides.learning_rate.log is True
    assert config.search_space_overrides.batch_size_low_res == [16, 32, 48, 64]


@pytest.mark.unit
def test_search_space_overrides_custom():
    """Test OptunaConfig accepts custom SearchSpaceOverrides."""
    from orchard.core.config.optuna_config import FloatRange, SearchSpaceOverrides

    custom_ov = SearchSpaceOverrides(
        learning_rate=FloatRange(low=1e-4, high=1e-1, log=True),
        batch_size_high_res=[4, 8],
    )
    config = OptunaConfig(search_space_overrides=custom_ov)

    assert config.search_space_overrides.learning_rate.low == pytest.approx(1e-4)
    assert config.search_space_overrides.learning_rate.high == pytest.approx(1e-1)
    assert config.search_space_overrides.batch_size_high_res == [4, 8]


@pytest.mark.unit
def test_search_space_overrides_from_dict():
    """Test SearchSpaceOverrides can be constructed from nested dict (YAML-style)."""
    config = OptunaConfig(
        search_space_overrides={
            "learning_rate": {"low": 1e-3, "high": 0.5, "log": True},
            "dropout": {"low": 0.2, "high": 0.7},
        }
    )

    assert config.search_space_overrides.learning_rate.low == pytest.approx(1e-3)
    assert config.search_space_overrides.learning_rate.high == pytest.approx(0.5)
    assert config.search_space_overrides.dropout.low == pytest.approx(0.2)
    assert config.search_space_overrides.dropout.high == pytest.approx(0.7)
    # Non-overridden fields keep defaults
    assert config.search_space_overrides.momentum.low == pytest.approx(0.85)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
