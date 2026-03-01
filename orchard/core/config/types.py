"""
Semantic type Definitions & Validation Primitives.

Foundational type-system for the configuration engine. Leverages Pydantic's
Annotated types and Functional Validators to enforce domain-specific
constraints (probability ranges, learning rate boundaries, path integrity)
before reaching orchestration logic.

Core Responsibilities:
    * Path sanitization: Resolves paths to absolute forms with home directory
      expansion (~), ensuring consistency without disk I/O
    * Boundary enforcement: Strict validation of hyperparameters (LR,
      probabilities, smoothing) using field constraints to prevent unstable states
    * type aliasing: Centralized registry of domain-specific types (BatchSize,
      ProjectSlug, LearningRate) for semantic consistency
    * Serialization policy: Custom serialization for complex objects (Path)
      ensuring JSON/YAML compatibility

Catches invalid states at application edge during schema initialization,
preventing runtime failures in deeper orchestration layers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import AfterValidator, Field, PlainSerializer


# VALIDATORS
def _sanitize_path(v: str | Path) -> Path:
    """
    Resolve path to absolute form without disk side-effects.

    Expands user home directory (~) and converts to absolute path
    for consistency across environments. No filesystem validation
    is performed to avoid I/O during schema initialization.

    Args:
        v: Path object or string to sanitize

    Returns:
        Absolute Path with home directory expanded
    """
    return Path(v).expanduser().resolve()


# GENERIC PRIMITIVES
PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveFloat = Annotated[float, Field(gt=0.0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]
Probability = Annotated[float, Field(ge=0.0, le=1.0)]

# FILESYSTEM
ValidatedPath = Annotated[
    Path,
    AfterValidator(_sanitize_path),
    PlainSerializer(lambda v: str(v), when_used="json", return_type=str),
]

# TRAINING
BatchSize = Annotated[int, Field(ge=1, le=128)]

# MODEL GEOMETRY
ImageSize = Annotated[int, Field(ge=28, le=1024)]
Channels = Annotated[int, Field(ge=1, le=4)]
DropoutRate = Annotated[float, Field(ge=0.0, le=0.9)]

# OPTIMIZATION
LearningRate = Annotated[float, Field(gt=1e-8, lt=1.0)]
WeightDecay = Annotated[float, Field(ge=0.0, le=0.2)]
Momentum = Annotated[float, Field(ge=0.0, lt=1.0)]
SmoothingValue = Annotated[float, Field(ge=0.0, le=0.3)]
GradNorm = Annotated[float, Field(ge=0.0, le=100.0)]

# AUGMENTATIONS & TTA
RotationDegrees = Annotated[int, Field(ge=0, le=360)]
ZoomScale = Annotated[float, Field(gt=0.0, le=2.0)]
PixelShift = Annotated[float, Field(ge=0.0, le=50.0)]
BlurSigma = Annotated[float, Field(ge=0.0, le=5.0)]
KernelSize = Annotated[int, Field(ge=1, le=31)]

# SYSTEM & METADATA
ProjectSlug = Annotated[str, Field(pattern=r"^[a-z0-9_-]+$", min_length=3, max_length=50)]
LogFrequency = Annotated[int, Field(ge=1, le=1000)]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
DeviceType = Literal["auto", "cpu", "cuda", "mps"]
