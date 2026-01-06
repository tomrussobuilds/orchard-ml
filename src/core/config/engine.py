"""
MedMNIST Configuration & Orchestration Engine.

This module acts as the declarative core of the pipeline, defining the 
hierarchical schema and validation logic required to drive the Orchestrator. 
It leverages Pydantic to transform raw inputs (CLI, YAML) into a structured, 
type-safe manifest that synchronizes hardware state with experiment logic.

Key Architectural Features:
    * Hierarchical Aggregation: Unifies specialized sub-configs (System, Dataset, 
      Model, Training, Evaluation, Augmentation) into a single immutable object.
    * Cross-Domain Validation: Implements complex logic checks (e.g., AMP vs. 
      Device, LR bounds, Mixup scheduling) that span multiple sub-modules.
    * Metadata-Driven Injection: Centralizes the resolution of MedMNIST registry 
      properties, ensuring architectural synchronization across the entire stack.
    * Factory Polymorphism: Provides dual entry points for instantiation via 
      structured YAML files or dynamic CLI arguments.

By enforcing strict validation during the initialization phase, the engine 
guarantees that the RootOrchestrator operates within a logically sound 
and reproducible execution context.
"""

# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
import argparse
from pathlib import Path

# =========================================================================== #
#                                Third-Party Imports                          #
# =========================================================================== #
from pydantic import (
    BaseModel, ConfigDict, Field, model_validator
)

# =========================================================================== #
#                               Internal Imports                              #
# =========================================================================== #
from .system_config import SystemConfig
from .training_config import TrainingConfig
from .augmentation_config import AugmentationConfig
from .dataset_config import DatasetConfig
from .evaluation_config import EvaluationConfig
from .models_config import ModelConfig

from ..metadata import DATASET_REGISTRY
from ..io import load_config_from_yaml
from ..paths import PROJECT_ROOT

# =========================================================================== #
#                                MAIN CONFIGURATION                          #
# =========================================================================== #

class Config(BaseModel):
    """
    Main Experiment Manifest and Orchestration Schema.
    
    Aggregates specialized sub-configurations into a single validated object.
    It provides the blueprint for the RootOrchestrator to execute experiments.
    """
    model_config = ConfigDict(
            extra="forbid",
            validate_assignment=True,
            frozen=True
    )
    
    system: SystemConfig = Field(default_factory=SystemConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)

    def dump_portable(self) -> dict:
        """
        Serializes the entire configuration while ensuring all filesystem 
        paths (system and dataset) are anchored to PROJECT_ROOT.
        """
        full_data = self.model_dump()

        full_data["system"] = self.system.to_portable_dict()    

        if "dataset" in full_data and "data_root" in full_data["dataset"]:
            dr_path = Path(full_data["dataset"]["data_root"])
            if dr_path.is_relative_to(PROJECT_ROOT):
                relative_dr = dr_path.relative_to(PROJECT_ROOT)
                full_data["dataset"]["data_root"] = f"./{relative_dr}"
            
        return full_data    
    

    @model_validator(mode="after")
    def validate_logic(self) -> "Config":
        """
        Cross-field logic validation after instantiation.
        Ensures hardware, model, and dataset parameters are logically aligned.
        """
        # 1. Training Logic
        if self.training.mixup_epochs > self.training.epochs:
            raise ValueError(
                f"mixup_epochs ({self.training.mixup_epochs}) cannot exceed "
                f"total epochs ({self.training.epochs})."
            )
            
        # 2. Hardware vs Feature alignment
        if self.system.device == "cpu" and self.training.use_amp:
            raise ValueError("AMP cannot be enabled when using CPU device.")
            
        # 3. Model vs Dataset consistency (Sfrutta le nuove properties)
        if self.model.pretrained and self.dataset.in_channels != 3:
            raise ValueError(
                f"Pretrained {self.model.name} require 3-channel input. "
                f"Current configuration provides {self.model.in_channels}."
            )
            
        # 4. Optimizer bounds
        if self.training.min_lr >= self.training.learning_rate:
            raise ValueError(
                f"min_lr ({self.training.min_lr}) must be less than "
                f"initial learning_rate ({self.training.learning_rate})."
            )
        return self
    
    def dump_serialized(self) -> dict:
        """
        Converts the config into a JSON-compatible dictionary.
        Essential for saving 'config.yaml' without Path-type errors.
        """
        return self.model_dump(mode="json")

    @property
    def run_slug(self) -> str:
        """Unique identifier for the experiment folder."""
        return f"{self.dataset.dataset_name}_{self.model.name}"
    
    @property
    def num_workers(self) -> int:
        return self.system.effective_num_workers
    
    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "Config":
        """Factory method to create a validated Config instance from a YAML file."""
        raw_data = load_config_from_yaml(yaml_path)
        return cls(**raw_data)        
    
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "Config":
        """
        Unified entry point for CLI-based instantiation.
        
        Orchestrates the resolution of dataset metadata and ensures that 
        channel promotion logic is synchronized between Dataset and Model.
        """
        if getattr(args, 'config', None):
            return cls.from_yaml(Path(args.config))

        # 1. Resolve Dataset Identity without hardcoding
        ds_name_raw = getattr(args, 'dataset', None)
        if not ds_name_raw:
            raise ValueError(
                "Dataset name must be provided via --dataset or a config file."
            )
            
        ds_name = ds_name_raw.lower()
        if ds_name not in DATASET_REGISTRY:
             raise ValueError(f"Dataset '{ds_name_raw}' not found in registry.")
        
        ds_meta = DATASET_REGISTRY[ds_name]

        # 2. Sequential Injection 
        ds_config = DatasetConfig.from_args(args, metadata=ds_meta)
        
        # We ensure ModelConfig is aware of the final channel decision
        model_config = ModelConfig.from_args(args, metadata=ds_meta)

        # 3. Final Aggregation
        return cls(
            system=SystemConfig.from_args(args),
            training=TrainingConfig.from_args(args),
            augmentation=AugmentationConfig.from_args(args),
            dataset=ds_config,
            model=model_config,
            evaluation=EvaluationConfig.from_args(args)
        )