"""
Semantic Type Definitions & Validation Primitives.

This module acts as the foundational type-system for the configuration engine. 
It leverages Pydantic's Annotated types and Functional Validators to enforce 
domain-specific constraints (e.g., physical probability ranges, learning rate 
boundaries, and path integrity) before they reach the orchestration logic.

Core Responsibilities:
    * Path Sanitization: Automatic expansion and creation of experiment 
      directories via `_ensure_dir`.
    * Boundary Enforcement: Strict validation of hyperparameters (Learning Rates, 
      Probabilities, Smoothing Values) using LaTeX-style interval logic.
    * Reusability: Provides a centralized registry of Type Aliases used across 
      all sub-configuration modules (System, Training, Dataset).

By centralizing these definitions, the engine ensures that invalid states are 
caught at the 'edge' of the application (CLI/YAML parsing) rather than 
during active training execution.
"""

# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
import os
from pathlib import Path
from typing import Annotated, Literal

# =========================================================================== #
#                                Third-Party Imports                          #
# =========================================================================== #
from pydantic import (
    Field, AfterValidator, PlainSerializer
)

# =========================================================================== #
#                                VALIDATORS                                   #
# =========================================================================== #

def _sanitize_path(v: Path) -> Path:
    """Resolve path to absolute form without disk side-effects."""
    return v.expanduser().resolve()

# =========================================================================== #
#                                1. GENERIC PRIMITIVES                        #
# =========================================================================== #

PositiveInt      = Annotated[int, Field(gt=0)]
NonNegativeInt   = Annotated[int, Field(ge=0)]
PositiveFloat    = Annotated[float, Field(gt=0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]
Percentage       = Annotated[float, Field(gt=0.0, le=1.0)]
Probability      = Annotated[float, Field(ge=0.0, le=1.0)]

# =========================================================================== #
#                                2. FILESYSTEM                                #
# =========================================================================== #

ValidatedPath = Annotated[
    Path,
    AfterValidator(_sanitize_path),
    PlainSerializer(lambda v: str(v), when_used="json", return_type=str)
]

# =========================================================================== #
#                                3. HARDWARE & PERFORMANCE                    #
# =========================================================================== #

WorkerCount = Annotated[int, Field(ge=0)]
BatchSize   = Annotated[int, Field(ge=1, le=2048)]

# =========================================================================== #
#                                4. MODEL GEOMETRY                            #
# =========================================================================== #

ImageSize   = Annotated[int, Field(ge=28, le=1024)]
Channels    = Annotated[int, Field(ge=1, le=4)]
DropoutRate = Annotated[float, Field(ge=0.0, le=0.9)]

# =========================================================================== #
#                                5. OPTIMIZATION                              #
# =========================================================================== #

LearningRate   = Annotated[float, Field(gt=1e-8, lt=1.0)]
WeightDecay    = Annotated[float, Field(ge=0.0, le=0.2)]
Momentum       = Annotated[float, Field(ge=0.0, lt=1.0)]
SmoothingValue = Annotated[float, Field(ge=0.0, le=0.3)]
GradNorm       = Annotated[float, Field(ge=0.0, le=100.0)]

# =========================================================================== #
#                                6. AUGMENTATIONS & TTA                       #
# =========================================================================== #

RotationDegrees = Annotated[int, Field(ge=0, le=360)]
ZoomScale       = Annotated[float, Field(gt=0.0, le=2.0)]
PixelShift      = Annotated[float, Field(ge=0.0, le=50.0)]
BlurSigma       = Annotated[float, Field(ge=0.0, le=5.0)]

# =========================================================================== #
#                                7. SYSTEM & METADATA                         #
# =========================================================================== #

ProjectSlug  = Annotated[
    str,
    Field(pattern=r"^[a-z0-9_-]+$", min_length=3, max_length=50)
]
LogFrequency = Annotated[int, Field(ge=1, le=1000)]
LogLevel     = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]