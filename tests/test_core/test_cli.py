"""
Smoke Tests for CLI Argument Parsing Module.

Tests command-line argument parsing and validation.
These are essential smoke tests to boost coverage from 8.79% to ~25%.
"""

# =========================================================================== #
#                         Standard Imports                                    #
# =========================================================================== #
from unittest.mock import patch

# =========================================================================== #
#                         Third-Party Imports                                 #
# =========================================================================== #
import pytest

# =========================================================================== #
#                         Internal Imports                                    #
# =========================================================================== #
from orchard.core.cli import parse_args

# =========================================================================== #
#                    PARSE ARGS: BASIC TESTS                                  #
# =========================================================================== #


@pytest.mark.unit
def test_parse_args_defaults():
    """Test parse_args returns default values when no arguments provided."""
    with patch("sys.argv", ["prog"]):
        args = parse_args()

        # Verify some key defaults
        assert args.dataset == "bloodmnist"
        assert args.epochs == 60
        assert args.batch_size == 16
        assert args.resolution == 28
        assert args.model_name == "resnet_18_adapted"


@pytest.mark.unit
def test_parse_args_dataset_selection():
    """Test parse_args handles dataset selection."""
    with patch("sys.argv", ["prog", "--dataset", "pathmnist"]):
        args = parse_args()

        assert args.dataset == "pathmnist"


@pytest.mark.unit
def test_parse_args_training_hyperparameters():
    """Test parse_args handles training hyperparameters."""
    with patch("sys.argv", ["prog", "--epochs", "100", "--batch_size", "64", "--lr", "0.001"]):
        args = parse_args()

        assert args.epochs == 100
        assert args.batch_size == 64
        assert args.lr == 0.001


@pytest.mark.unit
def test_parse_args_device_selection():
    """Test parse_args handles device selection."""
    with patch("sys.argv", ["prog", "--device", "cpu"]):
        args = parse_args()

        assert args.device == "cpu"


@pytest.mark.unit
def test_parse_args_config_file():
    """Test parse_args handles config file path."""
    with patch("sys.argv", ["prog", "--config", "path/to/config.yaml"]):
        args = parse_args()

        assert args.config == "path/to/config.yaml"


# =========================================================================== #
#                    PARSE ARGS: BOOLEAN FLAGS                                #
# =========================================================================== #


@pytest.mark.unit
def test_parse_args_reproducible_flag():
    """Test parse_args handles reproducible flag."""
    with patch("sys.argv", ["prog", "--reproducible"]):
        args = parse_args()

        assert args.reproducible is True


@pytest.mark.unit
def test_parse_args_use_amp_flag():
    """Test parse_args handles AMP flag."""
    with patch("sys.argv", ["prog", "--use_amp"]):
        args = parse_args()

        assert args.use_amp is True


@pytest.mark.unit
def test_parse_args_no_amp_flag():
    """Test parse_args handles no_amp flag."""
    with patch("sys.argv", ["prog", "--no_amp"]):
        args = parse_args()

        assert args.use_amp is False


@pytest.mark.unit
def test_parse_args_pretrained_flags():
    """Test parse_args handles pretrained flags."""
    # Default is True
    with patch("sys.argv", ["prog"]):
        args = parse_args()
        assert args.pretrained is True

    # Explicit --pretrained
    with patch("sys.argv", ["prog", "--pretrained"]):
        args = parse_args()
        assert args.pretrained is True

    # --no_pretrained
    with patch("sys.argv", ["prog", "--no_pretrained"]):
        args = parse_args()
        assert args.pretrained is False


@pytest.mark.unit
def test_parse_args_tta_flags():
    """Test parse_args handles TTA flags."""
    # Default
    with patch("sys.argv", ["prog"]):
        args = parse_args()
        assert hasattr(args, "use_tta")

    # --no_tta
    with patch("sys.argv", ["prog", "--no_tta"]):
        args = parse_args()
        assert args.use_tta is False


# =========================================================================== #
#                    PARSE ARGS: RESOLUTION                                   #
# =========================================================================== #


@pytest.mark.unit
def test_parse_args_resolution_28():
    """Test parse_args handles resolution 28."""
    with patch("sys.argv", ["prog", "--resolution", "28"]):
        args = parse_args()

        assert args.resolution == 28


@pytest.mark.unit
def test_parse_args_resolution_224():
    """Test parse_args handles resolution 224."""
    with patch("sys.argv", ["prog", "--resolution", "224"]):
        args = parse_args()

        assert args.resolution == 224


# =========================================================================== #
#                    PARSE ARGS: SCHEDULER TYPES                              #
# =========================================================================== #


@pytest.mark.unit
@pytest.mark.parametrize(
    "scheduler_type", ["cosine", "plateau", "step", "none"]
)
def test_parse_args_scheduler_types(scheduler_type):
    """Test parse_args handles all scheduler types."""
    with patch("sys.argv", ["prog", "--scheduler_type", scheduler_type]):
        args = parse_args()

        assert args.scheduler_type == scheduler_type


# =========================================================================== #
#                    PARSE ARGS: CRITERION TYPES                              #
# =========================================================================== #


@pytest.mark.unit
@pytest.mark.parametrize(
    "criterion_type", ["cross_entropy", "bce_logit", "focal"]
)
def test_parse_args_criterion_types(criterion_type):
    """Test parse_args handles all criterion types."""
    with patch("sys.argv", ["prog", "--criterion_type", criterion_type]):
        args = parse_args()

        assert args.criterion_type == criterion_type


# =========================================================================== #
#                    PARSE ARGS: OPTUNA PARAMETERS                            #
# =========================================================================== #


@pytest.mark.unit
def test_parse_args_optuna_n_trials():
    """Test parse_args handles Optuna n_trials."""
    with patch("sys.argv", ["prog", "--n_trials", "100"]):
        args = parse_args()

        assert args.n_trials == 100


@pytest.mark.unit
def test_parse_args_optuna_metric():
    """Test parse_args handles Optuna metric selection."""
    with patch("sys.argv", ["prog", "--metric_name", "accuracy"]):
        args = parse_args()

        assert args.metric_name == "accuracy"


@pytest.mark.unit
@pytest.mark.parametrize(
    "sampler_type", ["tpe", "cmaes", "random", "grid"]
)
def test_parse_args_optuna_sampler_types(sampler_type):
    """Test parse_args handles all Optuna sampler types."""
    with patch("sys.argv", ["prog", "--sampler_type", sampler_type]):
        args = parse_args()

        assert args.sampler_type == sampler_type


@pytest.mark.unit
@pytest.mark.parametrize(
    "pruner_type", ["median", "percentile", "hyperband", "none"]
)
def test_parse_args_optuna_pruner_types(pruner_type):
    """Test parse_args handles all Optuna pruner types."""
    with patch("sys.argv", ["prog", "--pruner_type", pruner_type]):
        args = parse_args()

        assert args.pruner_type == pruner_type


@pytest.mark.unit
def test_parse_args_optuna_direction():
    """Test parse_args handles Optuna optimization direction."""
    with patch("sys.argv", ["prog", "--direction", "minimize"]):
        args = parse_args()

        assert args.direction == "minimize"


# =========================================================================== #
#                    PARSE ARGS: AUGMENTATION                                 #
# =========================================================================== #


@pytest.mark.unit
def test_parse_args_augmentation_params():
    """Test parse_args handles augmentation parameters."""
    with patch(
        "sys.argv",
        [
            "prog",
            "--hflip", "0.5",
            "--rotation_angle", "15",
            "--jitter_val", "0.3",
        ],
    ):
        args = parse_args()

        assert args.hflip == 0.5
        assert args.rotation_angle == 15
        assert args.jitter_val == 0.3


# =========================================================================== #
#                    PARSE ARGS: PATHS                                        #
# =========================================================================== #


@pytest.mark.unit
def test_parse_args_paths():
    """Test parse_args handles path arguments."""
    with patch(
        "sys.argv",
        [
            "prog",
            "--data_dir", "/path/to/data",
            "--output_dir", "/path/to/outputs",
        ],
    ):
        args = parse_args()

        assert args.data_dir == "/path/to/data"
        assert args.output_dir == "/path/to/outputs"


# =========================================================================== #
#                    PARSE ARGS: EVALUATION                                   #
# =========================================================================== #


@pytest.mark.unit
def test_parse_args_evaluation_params():
    """Test parse_args handles evaluation parameters."""
    with patch(
        "sys.argv",
        [
            "prog",
            "--n_samples", "20",
            "--fig_dpi", "300",
            "--report_format", "json",
        ],
    ):
        args = parse_args()

        assert args.n_samples == 20
        assert args.fig_dpi == 300
        assert args.report_format == "json"


# =========================================================================== #
#                    PARSE ARGS: COMPLEX COMBINATIONS                         #
# =========================================================================== #


@pytest.mark.unit
def test_parse_args_complex_combination():
    """Test parse_args handles multiple arguments together."""
    with patch(
        "sys.argv",
        [
            "prog",
            "--dataset", "organcmnist",
            "--model_name", "efficientnet_b0",
            "--epochs", "50",
            "--batch_size", "128",
            "--lr", "0.0001",
            "--device", "cuda",
            "--resolution", "224",
            "--no_pretrained",
            "--use_amp",
        ],
    ):
        args = parse_args()

        assert args.dataset == "organcmnist"  # Fixed
        assert args.model_name == "efficientnet_b0"
        assert args.epochs == 50
        assert args.batch_size == 128
        assert args.lr == 0.0001
        assert args.device == "cuda"
        assert args.resolution == 224
        assert args.pretrained is False
        assert args.use_amp is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
