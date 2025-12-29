"""
MedMNIST Configuration & Orchestration Engine.

This module acts as the declarative core of the pipeline, defining the 
hierarchical schema and validation logic required to drive the Orchestrator. 
It leverages Pydantic to transform raw inputs (CLI, YAML) into a structured, 
type-safe manifest that synchronizes hardware state with experiment logic.

Key Architectural Components:
    * Configuration Aggregator: Centralizes specialized sub-configs (System, 
      Training, etc.) into a single, immutable experiment manifest.
    * Integrity Layer: Implements cross-field validation to ensure logical 
      alignment between hardware, model, and dataset.
    * Multi-Source Factory: Provides unified entry points for instantiation 
      via YAML manifests or CLI argparse namespaces.
"""

# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
import os
import argparse
from pathlib import Path

# =========================================================================== #
#                                Third-Party Imports                          #
# =========================================================================== #
from pydantic import (
    BaseModel, ConfigDict, Field, field_validator, model_validator
)

# =========================================================================== #
#                               Internal Imports                              #
# =========================================================================== #
from .types import NonNegativeInt
from .system_config import SystemConfig
from .training_config import TrainingConfig
from .augmentation_config import AugmentationConfig
from .dataset_config import DatasetConfig
from .evaluation_config import EvaluationConfig
from .models_config import ModelConfig

from ..environment import get_num_workers
from ..io import load_config_from_yaml

# =========================================================================== #
#                                MAIN CONFIGURATION                          #
# =========================================================================== #

class Config(BaseModel):
    """
    Main Experiment Manifest and Orchestration Schema.
    
    This class serves as the central source of truth for the entire pipeline,
    aggregating specialized sub-configurations into a single validated object.
    It acts as the primary configuration provider for the Orchestrator, 
    ensuring that hardware settings, dataset constraints, and optimization 
    strategies are logically aligned before execution.

    Key Responsibilities:
        * State Validation: Enforces cross-module consistency (e.g., matching 
          model input channels with dataset transformations).
        * Resource Negotiation: Bridges system capabilities (CPU/GPU/AMP) 
          with training requirements.
        * Orchestrator Interface: Provides a structured, immutable contract 
          that the Orchestrator uses to initialize data loaders, model 
          architectures, and logging utilities.
    """
    model_config = ConfigDict(
            extra="forbid",
            validate_assignment=True,
            frozen=True
    )
    
    # Nested configurations - Explicit access required (e.g., cfg.training.seed)
    system: SystemConfig = Field(default_factory=SystemConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)

    num_workers: NonNegativeInt = Field(default_factory=get_num_workers)


    @model_validator(mode="after")
    def validate_logic(self) -> "Config":
        """Cross-field logic validation after instantiation."""
        if self.training.mixup_epochs > self.training.epochs:
            raise ValueError(
                f"mixup_epochs ({self.training.mixup_epochs}) cannot exceed "
                f"epochs ({self.training.epochs})."
            )
        is_cpu = self.system.device == "cpu"
        if is_cpu and self.training.use_amp:
            raise ValueError("AMP cannot be enabled when using CPU device.")
        if self.model.pretrained and self.dataset.in_channels == 1 and not self.dataset.force_rgb:
            raise ValueError(
                "Pretrained models require 3-channel input. "
                "Set force_rgb=True in dataset config."
            )
        if self.training.min_lr >= self.training.learning_rate:
            raise ValueError(
                f"min_lr ({self.training.min_lr}) must be less than "
                f"learning_rate ({self.training.learning_rate})."
            )
        return self

    @field_validator("num_workers")
    @classmethod
    def check_cpu_count(cls, v: int) -> int:
        cpu_count = os.cpu_count() or 1
        return min(v, cpu_count)
    
    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "Config":
        """
        Factory method to create a validated Config instance from a YAML file.
        """
        raw_data = load_config_from_yaml(yaml_path)
        return cls(**raw_data)        
            
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "Config":
        """
        Factory method to create a validated Config instance from a CLI namespace.
        
        This method orchestrates the transformation of raw argparse parameters into 
        the hierarchical Pydantic schema, applying conditional logic for hardware 
        compatibility and dataset metadata resolution.
        """
        
        # 1. Short-circuit: If a --config YAML is provided, load directly from it
        if hasattr(args, 'config') and args.config:
            return cls.from_yaml(Path(args.config))
        
        # --- LOGIC EXECUTION & ASSEMBLY ---

        return cls(
            num_workers=getattr(args, 'num_workers', 4),
            system=SystemConfig.from_args(args),
            training=TrainingConfig.from_args(args),
            augmentation=AugmentationConfig.from_args(args),
            dataset=DatasetConfig.from_args(args),
            evaluation=EvaluationConfig.from_args(args),
            model=ModelConfig.from_args(args)
        )