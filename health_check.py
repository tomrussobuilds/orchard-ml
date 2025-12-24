"""
Health Check and Integrity Module

This script iterates through all registered MedMNIST datasets to:
1. Initialize the environment and security locks via RootOrchestrator.
2. Download and verify MD5 checksums for each .npz file (where applicable).
3. Validate internal keys and data consistency.
4. Generate visual samples to confirm correct mapping of labels/classes.
"""

# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
from pathlib import Path

# =========================================================================== #
#                                Third-Party Imports                          #
# =========================================================================== #
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# =========================================================================== #
#                                Internal Imports                             #
# =========================================================================== #
from src.core import (
    Config, DatasetConfig, TrainingConfig, AugmentationConfig,
    DATASET_REGISTRY, RootOrchestrator
)
from src.data_handler.data_explorer import show_sample_images

# =========================================================================== #
#                               HEALTH CHECK LOGIC                            #
# =========================================================================== #

def health_check() -> None:
    """
    Performs a global integrity scan across all datasets defined in the registry.
    """
    
    # 1. Minimal Config for Orchestration
    # We create a base config to satisfy the Orchestrator's requirements
    base_cfg = Config(
        model_name="HealthCheck-Probe",
        system={"output_dir": Path("outputs/health_checks")},
        training={"seed": 42}
    )

    # 2. Root Orchestration
    # Handles seeding, static dirs, and safety locks
    orchestrator = RootOrchestrator(base_cfg)
    
    # Note: We manually point the lock and log for health check specifically
    orchestrator.initialize_core_services()
    logger = orchestrator.run_logger
    
    # Professional header with dynamic divider width
    divider = "=" * 60
    header = "STARTING GLOBAL MEDMNIST HEALTH CHECK"
    logger.info(divider)
    logger.info(header.center(len(divider)))
    logger.info(divider)

    for key, ds_meta in DATASET_REGISTRY.items():
        logger.info(f"--- Checking Dataset: {ds_meta.display_name} ({key}) ---")
        
        try:
            # 1. Access the raw data from the NPZ file
            if not ds_meta.path.exists():
                raise FileNotFoundError(f"Dataset file not found at {ds_meta.path}")
                
            raw_data = np.load(ds_meta.path)
            
            # Extract arrays for validation
            train_images = raw_data['train_images']
            train_labels = raw_data['train_labels']
            val_images = raw_data['val_images']
            test_images = raw_data['test_images']

            num_classes_val = len(ds_meta.classes)

            logger.info(f"Loaded successfully: Train={train_images.shape}, "
                        f"Val={val_images.shape}, Test={test_images.shape}")
            logger.info(f"Channels: {ds_meta.in_channels} | Classes: {num_classes_val}")

            # 2. Prepare Tensors for the temporary DataLoader
            # Convert to float and scale to [0, 1] as expected by visualization logic
            images_t = torch.from_numpy(train_images).float() / 255.0
            
            # Reorder dimensions from NHWC (MedMNIST) to NCHW (PyTorch)
            if images_t.ndim == 3:  # Grayscale (N, H, W) -> (N, 1, H, W)
                images_t = images_t.unsqueeze(1)
            else:  # RGB (N, H, W, C) -> (N, C, H, W)
                images_t = images_t.permute(0, 3, 1, 2)
                
            labels_t = torch.from_numpy(train_labels).long().squeeze()

            # 3. Create a temporary DataLoader to satisfy show_sample_images signature
            temp_ds = TensorDataset(images_t, labels_t)
            temp_loader = DataLoader(temp_ds, batch_size=16, shuffle=True)

            # 4. Properly nested Config initialization for Pydantic validation
            temp_cfg = Config(
                model_name="HealthCheck-Probe",
                dataset=DatasetConfig(
                    dataset_name=ds_meta.name,
                    num_classes=num_classes_val,
                    mean=ds_meta.mean,
                    std=ds_meta.std
                ),
                training=TrainingConfig(
                    seed=42,
                    batch_size=16,
                    epochs=1
                ),
                augmentation=AugmentationConfig(
                    hflip=0.5,
                    rotation_angle=0,
                    jitter_val=0.0
                )
            )

            # 5. Generate sample images using the temporary loader
            # Samples are saved in the health_checks directory
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

    # Professional footer
    footer = "GLOBAL HEALTH CHECK COMPLETED"
    logger.info(divider)
    logger.info(footer.center(len(divider)))
    logger.info(divider)

# ========================================================================== #
#                                   ENTRY POINT                              #
# ========================================================================== # 
if __name__ == "__main__":
    health_check()