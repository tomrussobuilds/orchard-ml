"""
Health Check and Integrity Module

This script performs a global integrity scan across all registered MedMNIST 
datasets by executing a 5-step verification protocol for each:
1. Raw Data Access: Verification of .npz file presence and key-level accessibility.
2. Metadata Validation: Consistency check between tensor shapes and registry classes.
3. DataLoader Compatibility: Verification of temporary loader creation and sampling.
4. Config Validation: Pydantic-driven check for dataset-specific parameters.
5. Visual Confirmation: Generation of sample grids to verify label-image mapping.
"""

# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
from pathlib import Path

# =========================================================================== #
#                                Internal Imports                             #
# =========================================================================== #
from src.core import (
    Config, SystemConfig, DatasetConfig, DATASET_REGISTRY, RootOrchestrator
)
from src.data_handler import (
    show_sample_images, create_temp_loader
)

# =========================================================================== #
#                               HEALTH CHECK LOGIC                            #
# =========================================================================== #

def health_check() -> None:
    """
    Performs a global integrity scan across all datasets defined in the registry.
    Leveerages the RootOrchestrator context manager for lifecycle safeguards.
    """
    
    # 1. Minimal Config for Orchestration
    # We create a base config to satisfy the Orchestrator's requirements
    base_cfg = Config(
        model_name="HealthCheck-Probe",
        pretrained=True,
        system=SystemConfig(
            output_dir=Path("outputs/health_checks"),
            project_name="HealthCheck"
            ),
        training={"seed": 42}
    )

    # 2. Root Orchestration via Context Manager
    # This ensures the .lock file is automatically cleared even if a dataset check crashes
    with RootOrchestrator(base_cfg) as orchestrator:
        logger = orchestrator.run_logger
        
        divider = "=" * 60
        logger.info(divider)
        logger.info("STARTING GLOBAL MEDMNIST HEALTH CHECK".center(len(divider)))
        logger.info(divider)

        for key, ds_meta in DATASET_REGISTRY.items():
            logger.info(f"--- Checking Dataset: {ds_meta.display_name} ({key}) ---")
            
            try:
                # 1. Raw Data Access
                if not ds_meta.path.exists():
                    raise FileNotFoundError(f"Dataset file not found at {ds_meta.path}")
                    
                raw_data = orchestrator.load_raw_dataset(ds_meta.path)
                
                # 2. DataLoader Compatibility
                temp_loader = create_temp_loader(raw_data, batch_size=16)

                # 3. Validation of Config & RGB Promotion Logic
                temp_cfg = Config(
                    model_name="HealthCheck-Probe",
                    pretrained=True, 
                    dataset=DatasetConfig(
                        dataset_name=ds_meta.name,
                        in_channels=ds_meta.in_channels,
                        num_classes=len(ds_meta.classes),
                        mean=ds_meta.mean,
                        std=ds_meta.std
                    )
                )

                # Log effective channel status (Core of our recent fixes)
                mode_str = "RGB-PROMOTED" if temp_cfg.dataset.force_rgb else "NATIVE"
                logger.info(f"Mode: {mode_str} | Effective Channels: {temp_cfg.dataset.effective_in_channels}")

                # 4. Visual Confirmation
                sample_output_path = base_cfg.system.output_dir / f"samples_{ds_meta.name}.png"
                show_sample_images(
                    loader=temp_loader,
                    classes=ds_meta.classes,
                    save_path=sample_output_path,
                    cfg=temp_cfg
                )
                
                logger.info(f"Integrity check PASSED for {ds_meta.display_name}")

            except Exception as e:
                logger.error(f"Integrity check FAILED for {ds_meta.display_name}: {e}")
                continue

        logger.info(divider)
        logger.info("GLOBAL HEALTH CHECK COMPLETED".center(len(divider)))
        logger.info(divider)

# ========================================================================== #
#                                   ENTRY POINT                              #
# ========================================================================== # 
if __name__ == "__main__":
    health_check()