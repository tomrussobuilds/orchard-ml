"""
Test Suite for Semantic type Definitions.

Tests Pydantic annotated types and validators for domain-specific
constraints (paths, hyperparameters, probabilities).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from orchard.core.config.types import (
    BatchSize,
    BlurSigma,
    Channels,
    DeviceType,
    DropoutRate,
    GradNorm,
    ImageSize,
    LearningRate,
    LogFrequency,
    LogLevel,
    Momentum,
    NonNegativeFloat,
    NonNegativeInt,
    PixelShift,
    PositiveInt,
    Probability,
    ProjectSlug,
    RotationDegrees,
    SmoothingValue,
    ValidatedPath,
    WeightDecay,
    ZoomScale,
)


# GENERIC PRIMITIVES: POSITIVE INT
@pytest.mark.unit
def test_positive_int_valid():
    """Test PositiveInt accepts values > 0."""

    class Model(BaseModel):
        value: PositiveInt

    assert Model(value=1).value == 1
    assert Model(value=100).value == 100


@pytest.mark.unit
def test_positive_int_invalid():
    """Test PositiveInt rejects values <= 0."""

    class Model(BaseModel):
        value: PositiveInt

    with pytest.raises(ValidationError):
        Model(value=0)

    with pytest.raises(ValidationError):
        Model(value=-5)


# GENERIC PRIMITIVES: NON-NEGATIVE
@pytest.mark.unit
def test_non_negative_int_valid():
    """Test NonNegativeInt accepts values >= 0."""

    class Model(BaseModel):
        value: NonNegativeInt

    assert Model(value=0).value == 0
    assert Model(value=100).value == 100


@pytest.mark.unit
def test_non_negative_int_invalid():
    """Test NonNegativeInt rejects negative values."""

    class Model(BaseModel):
        value: NonNegativeInt

    with pytest.raises(ValidationError):
        Model(value=-1)


@pytest.mark.unit
def test_non_negative_float_valid():
    """Test NonNegativeFloat accepts values >= 0.0."""

    class Model(BaseModel):
        value: NonNegativeFloat

    assert Model(value=0.0).value == pytest.approx(0.0)
    assert Model(value=1.5).value == pytest.approx(1.5)


# PROBABILITIES AND PERCENTAGES
@pytest.mark.unit
def test_probability_bounds():
    """Test Probability accepts [0.0, 1.0]."""

    class Model(BaseModel):
        prob: Probability

    assert Model(prob=0.0).prob == pytest.approx(0.0)
    assert Model(prob=0.5).prob == pytest.approx(0.5)
    assert Model(prob=1.0).prob == pytest.approx(1.0)

    with pytest.raises(ValidationError):
        Model(prob=-0.1)

    with pytest.raises(ValidationError):
        Model(prob=1.5)


# FILESYSTEM: VALIDATED PATH
@pytest.mark.unit
def test_validated_path_expansion():
    """Test ValidatedPath expands ~ and resolves to absolute."""

    class Model(BaseModel):
        path: ValidatedPath

    model = Model(path=Path("~/test"))

    assert model.path.is_absolute()
    assert "~" not in str(model.path)


@pytest.mark.unit
def test_validated_path_resolution():
    """Test ValidatedPath resolves relative paths."""

    class Model(BaseModel):
        path: ValidatedPath

    model = Model(path=Path("./dataset"))

    assert model.path.is_absolute()


@pytest.mark.unit
def test_validated_path_json_serialization():
    """Test ValidatedPath serializes to string in JSON."""

    class Model(BaseModel):
        path: ValidatedPath

    model = Model(path=Path("/mock/test"))
    json_data = model.model_dump(mode="json")

    assert isinstance(json_data["path"], str)
    assert json_data["path"] == str(model.path)


# TRAINING: BATCH SIZE
@pytest.mark.unit
def test_batch_size_bounds():
    """Test BatchSize must be in [1, 128]."""

    class Model(BaseModel):
        batch: BatchSize

    assert Model(batch=1).batch == 1
    assert Model(batch=128).batch == 128

    with pytest.raises(ValidationError):
        Model(batch=0)

    with pytest.raises(ValidationError):
        Model(batch=256)


# MODEL GEOMETRY
@pytest.mark.unit
def test_image_size_bounds():
    """Test ImageSize must be in [28, 1024]."""

    class Model(BaseModel):
        size: ImageSize

    assert Model(size=28).size == 28
    assert Model(size=224).size == 224
    assert Model(size=1024).size == 1024

    with pytest.raises(ValidationError):
        Model(size=16)

    with pytest.raises(ValidationError):
        Model(size=2048)


@pytest.mark.unit
def test_channels_bounds():
    """Test Channels must be in [1, 4]."""

    class Model(BaseModel):
        ch: Channels

    assert Model(ch=1).ch == 1
    assert Model(ch=3).ch == 3
    assert Model(ch=4).ch == 4

    with pytest.raises(ValidationError):
        Model(ch=0)

    with pytest.raises(ValidationError):
        Model(ch=5)


@pytest.mark.unit
def test_dropout_rate_bounds():
    """Test DropoutRate must be in [0.0, 0.9]."""

    class Model(BaseModel):
        dropout: DropoutRate

    assert Model(dropout=0.0).dropout == pytest.approx(0.0)
    assert Model(dropout=0.5).dropout == pytest.approx(0.5)
    assert Model(dropout=0.9).dropout == pytest.approx(0.9)

    with pytest.raises(ValidationError):
        Model(dropout=-0.1)

    with pytest.raises(ValidationError):
        Model(dropout=1.0)


# OPTIMIZATION HYPERPARAMETERS
@pytest.mark.unit
def test_learning_rate_bounds():
    """Test LearningRate must be in (1e-8, 1.0)."""

    class Model(BaseModel):
        lr: LearningRate

    assert Model(lr=0.001).lr == pytest.approx(0.001)
    assert Model(lr=1e-7).lr == pytest.approx(1e-7)

    with pytest.raises(ValidationError):
        Model(lr=0.0)

    with pytest.raises(ValidationError):
        Model(lr=1.0)


@pytest.mark.unit
def test_weight_decay_bounds():
    """Test WeightDecay must be in [0.0, 0.2]."""

    class Model(BaseModel):
        wd: WeightDecay

    assert Model(wd=0.0).wd == pytest.approx(0.0)
    assert Model(wd=0.01).wd == pytest.approx(0.01)
    assert Model(wd=0.2).wd == pytest.approx(0.2)

    with pytest.raises(ValidationError):
        Model(wd=-0.01)

    with pytest.raises(ValidationError):
        Model(wd=0.5)


@pytest.mark.unit
def test_momentum_bounds():
    """Test Momentum must be in [0.0, 1.0)."""

    class Model(BaseModel):
        mom: Momentum

    assert Model(mom=0.0).mom == pytest.approx(0.0)
    assert Model(mom=0.9).mom == pytest.approx(0.9)

    with pytest.raises(ValidationError):
        Model(mom=1.0)


@pytest.mark.unit
def test_smoothing_value_bounds():
    """Test SmoothingValue must be in [0.0, 0.3]."""

    class Model(BaseModel):
        smooth: SmoothingValue

    assert Model(smooth=0.0).smooth == pytest.approx(0.0)
    assert Model(smooth=0.1).smooth == pytest.approx(0.1)
    assert Model(smooth=0.3).smooth == pytest.approx(0.3)

    with pytest.raises(ValidationError):
        Model(smooth=0.5)


@pytest.mark.unit
def test_grad_norm_bounds():
    """Test GradNorm must be in [0.0, 100.0]."""

    class Model(BaseModel):
        grad: GradNorm

    assert Model(grad=0.0).grad == pytest.approx(0.0)
    assert Model(grad=1.0).grad == pytest.approx(1.0)
    assert Model(grad=100.0).grad == pytest.approx(100.0)

    with pytest.raises(ValidationError):
        Model(grad=200.0)


# AUGMENTATION TYPES
@pytest.mark.unit
def test_rotation_degrees_bounds():
    """Test RotationDegrees must be in [0, 360]."""

    class Model(BaseModel):
        rot: RotationDegrees

    assert Model(rot=0).rot == 0
    assert Model(rot=180).rot == 180
    assert Model(rot=360).rot == 360

    with pytest.raises(ValidationError):
        Model(rot=-10)

    with pytest.raises(ValidationError):
        Model(rot=400)


@pytest.mark.unit
def test_zoom_scale_bounds():
    """Test ZoomScale must be in (0.0, 2.0]."""

    class Model(BaseModel):
        zoom: ZoomScale

    assert Model(zoom=0.5).zoom == pytest.approx(0.5)
    assert Model(zoom=2.0).zoom == pytest.approx(2.0)

    with pytest.raises(ValidationError):
        Model(zoom=0.0)

    with pytest.raises(ValidationError):
        Model(zoom=3.0)


@pytest.mark.unit
def test_pixel_shift_bounds():
    """Test PixelShift must be in [0.0, 50.0]."""

    class Model(BaseModel):
        shift: PixelShift

    assert Model(shift=0.0).shift == pytest.approx(0.0)
    assert Model(shift=10.0).shift == pytest.approx(10.0)
    assert Model(shift=50.0).shift == pytest.approx(50.0)

    with pytest.raises(ValidationError):
        Model(shift=100.0)


@pytest.mark.unit
def test_blur_sigma_bounds():
    """Test BlurSigma must be in [0.0, 5.0]."""

    class Model(BaseModel):
        blur: BlurSigma

    assert Model(blur=0.0).blur == pytest.approx(0.0)
    assert Model(blur=2.5).blur == pytest.approx(2.5)
    assert Model(blur=5.0).blur == pytest.approx(5.0)

    with pytest.raises(ValidationError):
        Model(blur=10.0)


# SYSTEM METADATA TYPES
@pytest.mark.unit
def test_project_slug_pattern():
    """Test ProjectSlug enforces lowercase alphanumeric + underscore/dash."""

    class Model(BaseModel):
        slug: ProjectSlug

    assert Model(slug="my-project").slug == "my-project"
    assert Model(slug="project_v2").slug == "project_v2"
    assert Model(slug="abc123").slug == "abc123"

    with pytest.raises(ValidationError):
        Model(slug="MyProject")

    with pytest.raises(ValidationError):
        Model(slug="my project")

    with pytest.raises(ValidationError):
        Model(slug="project@v1")


@pytest.mark.unit
def test_project_slug_length():
    """Test ProjectSlug length must be in [3, 50]."""

    class Model(BaseModel):
        slug: ProjectSlug

    assert Model(slug="abc").slug == "abc"
    assert Model(slug="a" * 50).slug == "a" * 50

    with pytest.raises(ValidationError):
        Model(slug="ab")  # Too short

    with pytest.raises(ValidationError):
        Model(slug="a" * 51)  # Too long


@pytest.mark.unit
def test_log_frequency_bounds():
    """Test LogFrequency must be in [1, 1000]."""

    class Model(BaseModel):
        freq: LogFrequency

    assert Model(freq=1).freq == 1
    assert Model(freq=100).freq == 100
    assert Model(freq=1000).freq == 1000

    with pytest.raises(ValidationError):
        Model(freq=0)

    with pytest.raises(ValidationError):
        Model(freq=2000)


@pytest.mark.unit
def test_log_level_literals():
    """Test LogLevel accepts only valid logging levels."""

    class Model(BaseModel):
        level: LogLevel

    for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
        assert Model(level=level).level == level

    with pytest.raises(ValidationError):
        Model(level="INVALID")


@pytest.mark.unit
def test_device_type_literals():
    """Test DeviceType accepts only valid device strings."""

    class Model(BaseModel):
        device: DeviceType

    for device in ["auto", "cpu", "cuda", "mps"]:
        assert Model(device=device).device == device

    with pytest.raises(ValidationError):
        Model(device="gpu")

    with pytest.raises(ValidationError):
        Model(device="tpu")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
