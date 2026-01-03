"""Unit tests for the Configuration Engine.

This module validates the hierarchical Pydantic schema, the CLI factory 
mapping, and the YAML serialization/deserialization cycle.
"""
# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# =========================================================================== #
#                                Third-Party Imports                          #
# =========================================================================== #
import pytest
from pydantic import ValidationError
import torch

# =========================================================================== #
#                                Internal Imports                             #
# =========================================================================== #
from src.core.config import Config, TrainingConfig
from src.core.orchestrator import RootOrchestrator
from src.core.io import save_config_as_yaml


# =========================================================================== #
#                                   FIXTURES                                  #
# =========================================================================== #

@pytest.fixture
def mock_args():
    """Provides a minimal argparse.Namespace object for testing."""
    args = argparse.Namespace()
    args.config = None
    args.dataset = "bloodmnist"
    args.model_name = "ResNet-18"
    args.pretrained = True
    args.num_workers = 2
    args.device = "cpu"
    args.data_dir = "./data"
    args.output_dir = "./outputs"
    args.save_model = True
    args.log_interval = 10
    args.seed = 42
    args.batch_size = 32
    args.lr = 0.001
    args.momentum = 0.9
    args.weight_decay = 1e-4
    args.epochs = 5
    args.patience = 3
    args.mixup_alpha = 0.2
    args.mixup_epochs = 2
    args.use_tta = True
    args.cosine_fraction = 0.5
    args.use_amp = False
    args.grad_clip = 1.0
    args.hflip = 0.5
    args.rotation_angle = 15
    args.jitter_val = 0.1
    args.project_name = "test_project"
    return args


# =========================================================================== #
#                                 TEST CASES                                  #
# =========================================================================== #

@pytest.mark.parametrize("field, value", [
    ("learning_rate", -0.01),
    ("learning_rate", 0.0),
    ("epochs", 0),
    ("momentum", 1.1),
    ("batch_size", -1),
])
def test_pydantic_boundary_constraints(field, value):
    """Tests if Pydantic raises errors for out-of-bounds hyperparameter values."""
    with pytest.raises(ValidationError):
        TrainingConfig(**{field: value})


def test_config_immutability(mock_args):
    """Verifies that the config object is strictly read-only (frozen)."""
    cfg = Config.from_args(mock_args)
    with pytest.raises(ValidationError):
        # Pydantic v2 raises ValidationError on assignment to frozen models
        cfg.training.learning_rate = 0.1


def test_mixup_epochs_logic_validation(mock_args):
    """Ensures cross-field validation: mixup_epochs cannot exceed total epochs."""
    mock_args.epochs = 10
    mock_args.mixup_epochs = 15
    
    with pytest.raises((ValueError, ValidationError), match="mixup_epochs .* cannot exceed total epochs"):
        Config.from_args(mock_args)


def test_amp_cpu_compatibility_validation(mock_args):
    """Ensures the orchestrator prevents AMP on CPU devices."""
    mock_args.device = "cpu"
    mock_args.use_amp = True
    with pytest.raises(ValueError, match="AMP cannot be enabled when using CPU"):
        Config.from_args(mock_args)


def test_yaml_roundtrip_integrity(tmp_path, mock_args):
    """Verifies that saving and loading a YAML preserves all configuration values."""
    original_cfg = Config.from_args(mock_args)
    yaml_path = tmp_path / "config.yaml"
    
    save_config_as_yaml(
        data=original_cfg.model_dump(mode='json'),
        yaml_path=yaml_path
    )
    
    loaded_cfg = Config.from_yaml(yaml_path)
    assert original_cfg.model_dump() == loaded_cfg.model_dump()


def test_yaml_short_circuit_priority(tmp_path, mock_args):
    """Ensures that providing a --config file overrides other CLI arguments."""
    yaml_path = tmp_path / "override.yaml"
    original_cfg = Config.from_args(mock_args)
    
    cfg_dict = original_cfg.model_dump(mode='json')
    cfg_dict['training']['batch_size'] = 999
    save_config_as_yaml(cfg_dict, yaml_path)
    
    mock_args.config = str(yaml_path)
    mock_args.batch_size = 10
    
    final_cfg = Config.from_args(mock_args)
    assert final_cfg.training.batch_size == 999


def test_orchestrator_lifecycle(mock_args, tmp_path):
    """Verifies orchestrator directory setup and resource management."""
    mock_args.output_dir = str(tmp_path / "outputs")
    mock_args.data_dir = str(tmp_path / "data")
    mock_args.project_name = f"test_run_{tmp_path.name}"
    
    cfg = Config.from_args(mock_args)
    
    with RootOrchestrator(cfg) as orchestrator:
        assert orchestrator.paths.root.exists()
        assert orchestrator.paths.logs.is_dir()
        assert orchestrator.paths.models.is_dir()
        
        config_path = orchestrator.paths.get_config_path()
        assert config_path.exists()
        
        assert isinstance(orchestrator.get_device(), torch.device)

    lock_path = cfg.system.lock_file_path
    assert not lock_path.exists()

if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))