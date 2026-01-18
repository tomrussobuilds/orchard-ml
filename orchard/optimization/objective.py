"""
Optuna Objective Function for Vision Pipeline.

Implements the objective function that Optuna optimizes, integrating
seamlessly with the existing ModelTrainer and Config architecture.

Preserves dataset metadata across trial config reconstruction.
"""
# =========================================================================== #
#                         Standard Imports                                    #
# =========================================================================== #
import logging
from typing import Dict, Any
import gc

# =========================================================================== #
#                         Third-Party Imports                                 #
# =========================================================================== #
import optuna
import torch

# =========================================================================== #
#                         Internal Imports                                    #
# =========================================================================== #
from orchard.core import Config, LOGGER_NAME
from orchard.data_handler import load_medmnist, get_dataloaders
from orchard.models import get_model
from orchard.trainer import (
    ModelTrainer, validate_epoch, train_one_epoch,
    get_optimizer, get_scheduler, get_criterion
)

logger = logging.getLogger(LOGGER_NAME)


# =========================================================================== #
#                          OBJECTIVE FUNCTION                                 #
# =========================================================================== #

class OptunaObjective:
    """
    Objective function wrapper for Optuna optimization.
    
    Encapsulates the training pipeline and exposes a metric for optimization.
    Handles trial pruning based on intermediate validation results.
    
    Critical Design:
        - Caches resolved dataset metadata in __init__
        - Re-injects metadata into trial configs (lost during serialization)
        - Loads MedMNISTData once and reuses across trials
    """
    
    def __init__(
        self,
        cfg: Config,
        search_space: Dict[str, Any],
        device: torch.device,
        metric_name: str = "auc",
        enable_pruning: bool = True,
        warmup_epochs: int = 10,
    ):
        """
        Initialize objective function.
        
        Args:
            cfg: Base Config to override with trial suggestions
            search_space: Dict of {param_name: suggest_function}
            device: PyTorch device for training
            metric_name: Metric to optimize (must be in validation metrics)
            enable_pruning: Enable early stopping of unpromising trials
            warmup_epochs: Minimum epochs before pruning activates
        """
        self.cfg = cfg
        self.search_space = search_space
        self.device = device
        self.metric_name = metric_name
        self.enable_pruning = enable_pruning
        self.warmup_epochs = warmup_epochs
        
        # CRITICAL FIX: Cache resolved metadata object
        # This ensures _ensure_metadata has been called and prevents None errors
        self.base_metadata = self.cfg.dataset._ensure_metadata
        
        # CRITICAL FIX: Load dataset once (returns MedMNISTData with path info)
        # This is shared across all trials to avoid repeated downloads
        self.medmnist_data = load_medmnist(self.base_metadata)
        
        logger.info(f"Objective initialized with metric: {metric_name}")
        logger.info(f"Cached dataset metadata: {self.base_metadata.name} "
                   f"({self.base_metadata.num_classes} classes)")
        logger.info(f"Dataset loaded from: {self.medmnist_data.path}")

    def __call__(self, trial: optuna.Trial) -> float:
        """
        Objective function called by Optuna for each trial.
        
        Args:
            trial: Optuna trial object for hyperparameter sampling
            
        Returns:
            Best validation metric achieved during training
        """
        # Sample hyperparameters
        if hasattr(self.search_space, "sample_params"):
            params = self.search_space.sample_params(trial)
        else:
            params = {
                key: fn(trial) for key, fn in self.search_space.items()
            }
        
        # Build trial-specific config (with metadata re-injected)
        trial_cfg = self._build_trial_config(params)
        
        # Log trial parameters
        logger.info(f"Trial {trial.number} params: {params}")
        
        # Create data loaders using cached MedMNISTData
        # Pass is_optuna=True to prevent worker leaks
        train_loader, val_loader, _ = get_dataloaders(
            self.medmnist_data,
            trial_cfg,
            is_optuna=True
        )
        
        # Initialize model (get_model returns model already on device)
        model = get_model(self.device, trial_cfg)
        
        # Initialize optimizer, scheduler, and criterion
        optimizer = get_optimizer(model, trial_cfg)
        scheduler = get_scheduler(optimizer, trial_cfg)
        criterion = get_criterion(trial_cfg)
        
        # Initialize trainer with all required components
        trainer = PrunableTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            cfg=trial_cfg,
            device=self.device
        )
        
        # Train with pruning
        try:
            best_metric = self._train_with_pruning(trainer, trial)
        finally:
            # Aggressive cleanup to prevent memory leaks
            del train_loader, val_loader, model, optimizer, scheduler, criterion, trainer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        
        return best_metric
    
    def _build_trial_config(self, trial_params: Dict[str, Any]) -> Config:
        """
        Create Config instance with trial-specific hyperparameters.
        
        CRITICAL: Re-injects cached metadata that gets excluded during
        model_dump() serialization.
        
        Args:
            trial_params: Sampled hyperparameters from Optuna
            
        Returns:
            Validated Config with trial parameters and metadata
        """
        config_dict = self.cfg.model_dump()
        
        # Ensure resolution is preserved
        if config_dict["dataset"].get("resolution") is None:
            config_dict["dataset"]["resolution"] = self.cfg.dataset.resolution
        
        # Re-inject cached metadata (excluded from serialization)
        config_dict["dataset"]["metadata"] = self.base_metadata
        
        # Override epochs with Optuna-specific value (shorter trials)
        config_dict["training"]["epochs"] = self.cfg.optuna.epochs
        
        # Apply trial-specific parameter overrides
        for param_name, value in trial_params.items():
            if param_name in ["learning_rate", "weight_decay", "momentum", "min_lr",
                              "mixup_alpha", "label_smoothing", "batch_size",
                              "cosine_fraction", "scheduler_patience"]:
                config_dict["training"][param_name] = value
            elif param_name == "dropout":
                config_dict["model"][param_name] = value
            elif param_name in ["rotation_angle", "jitter_val", "min_scale"]:
                config_dict["augmentation"][param_name] = value
        
        # Reconstruct Config (triggers Pydantic validation)
        return Config(**config_dict)
    
    def _train_with_pruning(
        self, 
        trainer: "PrunableTrainer", 
        trial: optuna.Trial
    ) -> float:
        """
        Execute training loop with Optuna pruning integration.
        
        Args:
            trainer: PrunableTrainer instance
            trial: Optuna trial for reporting and pruning
            
        Returns:
            Best validation metric achieved
            
        Raises:
            optuna.TrialPruned: If trial should be terminated early
        """
        best_metric = -float('inf')

        for epoch in range(1, trainer.epochs + 1):
            # Train one epoch
            epoch_loss = trainer.train_one_epoch_wrapper(epoch)

            # Validate
            val_metrics = trainer.validate_epoch_wrapper()

            # Extract target metric
            current_metric = self._extract_metric(val_metrics)
            best_metric = max(best_metric, current_metric)

            # Report to Optuna (for visualization and pruning)
            trial.report(current_metric, epoch)

            # Check if pruning should trigger
            if self.enable_pruning and epoch >= self.warmup_epochs:
                if trial.should_prune():
                    logger.info(
                        f"Trial {trial.number} pruned at epoch {epoch} "
                        f"({self.metric_name}={current_metric:.4f})"
                    )
                    raise optuna.TrialPruned()

            # Learning rate scheduling
            trainer._smart_step_scheduler(val_metrics["loss"])

            # Compact logging (every 5 epochs or at end)
            if epoch % 5 == 0 or epoch == trainer.epochs:
                logger.info(
                    f"T{trial.number} E{epoch}/{trainer.epochs} | "
                    f"Loss:{epoch_loss:.4f} | "
                    f"{self.metric_name}:{current_metric:.4f} "
                    f"(Best:{best_metric:.4f})"
                )

        return best_metric
    
    def _extract_metric(self, val_metrics: Dict[str, float]) -> float:
        """
        Extract target metric from validation results.
        
        Args:
            val_metrics: Dict of validation metrics
            
        Returns:
            Value of target metric
            
        Raises:
            KeyError: If metric_name not found in validation results
        """
        if self.metric_name not in val_metrics:
            available = list(val_metrics.keys())
            raise KeyError(
                f"Metric '{self.metric_name}' not found in validation results. "
                f"Available: {available}"
            )
        return val_metrics[self.metric_name]


# =========================================================================== #
#                          PRUNABLE TRAINER                                   #
# =========================================================================== #

class PrunableTrainer(ModelTrainer):
    """
    Extension of ModelTrainer with Optuna pruning support.
    
    Provides wrapper methods for epoch-level training and validation
    that integrate cleanly with Optuna's pruning callbacks.
    """

    def train_one_epoch_wrapper(self, epoch: int) -> float:
        """
        Train a single epoch and return average loss.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Average training loss for the epoch
        """
        # Apply MixUp only during first fraction of training
        mixup_cutoff = int(self.cfg.training.cosine_fraction * self.epochs)
        current_mixup = self.mixup_fn if epoch <= mixup_cutoff else None

        epoch_loss = train_one_epoch(
            model=self.model,
            loader=self.train_loader,
            criterion=self.criterion,
            optimizer=self.optimizer,
            device=self.device,
            mixup_fn=current_mixup,
            scaler=self.scaler,
            grad_clip=self.cfg.training.grad_clip,
            epoch=epoch,
            total_epochs=self.epochs,
            use_tqdm=False  # Disabled during optimization for cleaner logs
        )

        self.train_losses.append(epoch_loss)
        return epoch_loss

    def validate_epoch_wrapper(self) -> dict:
        """
        Validate one epoch and return metrics dictionary.
        
        Returns:
            Dict of validation metrics (loss, accuracy, auc, etc.)
        """
        val_metrics = validate_epoch(
            model=self.model,
            val_loader=self.val_loader,
            criterion=self.criterion,
            device=self.device
        )

        self.val_metrics_history.append(val_metrics)
        return val_metrics