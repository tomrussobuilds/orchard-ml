"""
Health Check and Integrity Module (Multi-Resolution with Visualization)

Performs integrity checks across MedMNIST datasets, supports 28x28 and 224x224 resolutions.
"""

# Standard Imports
import logging

# Internal Imports
from orchard.core import RootOrchestrator
from orchard.core.cli import parse_args
from orchard.core.config import Config
from orchard.core.metadata import DatasetRegistryWrapper
from orchard.data_handler.data_explorer import show_samples_for_dataset
from orchard.data_handler.factory import create_temp_loader
from orchard.data_handler.fetcher import load_medmnist_health_check

#                                Logging Setup                                #
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def health_check_single_dataset(ds_meta, orchestrator, resolution: int = 28) -> None:
    """
    Perform a health check on a single dataset: file presence, DataLoader, sample images.
    """
    run_logger = orchestrator.run_logger
    try:
        run_logger.info(
            f"Starting health check for dataset: {ds_meta.display_name} ({ds_meta.name})"
        )

        # ---------------- Step 1: Ensure dataset exists ----------------
        dataset_info = load_medmnist_health_check(ds_meta)
        dataset_path = getattr(dataset_info, "path", dataset_info)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found at {dataset_path}")
        run_logger.info(f"Dataset available at: {dataset_path}")

        # ---------------- Step 2: Build DataLoader ----------------
        loader = create_temp_loader(dataset_path, batch_size=32)

        # ---------------- Step 3: Sample batch ----------------
        batch_images, _ = next(iter(loader))
        run_logger.info(f"DataLoader test passed: batch size {len(batch_images)}")

        # ---------------- Step 4: Visual confirmation ----------------
        show_samples_for_dataset(
            loader=loader,
            classes=ds_meta.classes,
            dataset_name=ds_meta.name,
            run_paths=orchestrator.paths,
            resolution=resolution,
        )

        run_logger.info(f"Sample images saved for dataset: {ds_meta.display_name}")
        run_logger.info(f"Health check PASSED for dataset: {ds_meta.display_name}")

    except Exception as e:
        run_logger.error(
            f"Health check FAILED for dataset: {ds_meta.display_name} ({ds_meta.name})"
        )
        run_logger.exception(e)


def fetch_all_datasets_health_check() -> None:
    """
    Iterates over the MedMNIST dataset registry and performs a health check for each dataset.
    """
    args = parse_args()
    cfg = Config.from_args(args)

    with RootOrchestrator(cfg) as orchestrator:
        run_logger = orchestrator.run_logger
        run_logger.info("Starting health check for all MedMNIST datasets...")

        for key in ["28x28", "224x224"]:
            resolution = 28 if key == "28x28" else 224
            run_logger.info(f"Checking datasets with resolution: {resolution}x{resolution}")

            dataset_wrapper = DatasetRegistryWrapper(resolution=resolution)
            for _, ds_meta in dataset_wrapper.registry.items():
                try:
                    run_logger.info(f"Attempting health check for dataset '{ds_meta.name}'")
                    health_check_single_dataset(ds_meta, orchestrator, resolution=resolution)
                except Exception as e:
                    run_logger.warning(
                        f"Skipping dataset '{ds_meta.name}' due to error, but continuing."
                    )
                    run_logger.exception(e)

        run_logger.info("Health check completed for all datasets.")


# ENTRY POINT
if __name__ == "__main__":
    fetch_all_datasets_health_check()
