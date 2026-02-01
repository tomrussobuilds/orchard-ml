"""
Health Check and Integrity Module (Multi-Resolution with Visualization)
Performs integrity checks across MedMNIST datasets, supports 28x28 and 224x224 resolutions.
"""

import argparse
import logging

from orchard.core import RootOrchestrator
from orchard.core.config import Config
from orchard.core.metadata import DatasetRegistryWrapper
from orchard.data_handler.data_explorer import show_samples_for_dataset
from orchard.data_handler.fetcher import load_dataset_health_check
from orchard.data_handler.loader import create_temp_loader

# Logging Setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_health_check_args():
    """Parse command-line arguments for health check."""
    parser = argparse.ArgumentParser(description="Dataset health check utility")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Specific dataset to check (e.g., 'bloodmnist'). If not specified, checks all datasets.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        choices=[28, 224],
        help="Target resolution (28 or 224). If not specified, checks both resolutions.",
    )
    return parser.parse_args()


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
        dataset_info = load_dataset_health_check(ds_meta)
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
    args = parse_health_check_args()

    if args.resolution is not None:
        resolutions = [args.resolution]
    else:
        resolutions = [28, 224]

    from types import SimpleNamespace

    dummy_args = SimpleNamespace(
        config=None,
        dataset=args.dataset or "bloodmnist",
        resolution=28,
        model_name="mini_cnn",
        epochs=1,
        batch_size=32,
        max_samples=100,
        num_workers=0,
        no_amp=True,
        mixup_epochs=0,
        mixup_alpha=0.0,
    )

    for resolution in resolutions:
        dummy_args.resolution = resolution
        dummy_args.model_name = "mini_cnn" if resolution == 28 else "efficientnet_b0"

        cfg = Config.from_args(dummy_args)

        with RootOrchestrator(cfg) as orchestrator:
            run_logger = orchestrator.run_logger
            run_logger.info(f"Checking datasets with resolution: {resolution}x{resolution}")

            dataset_wrapper = DatasetRegistryWrapper(resolution=resolution)
            datasets_to_check = dataset_wrapper.registry.items()
            if args.dataset:
                datasets_to_check = [
                    (name, meta) for name, meta in datasets_to_check if name == args.dataset
                ]
                if not datasets_to_check:
                    run_logger.error(
                        f"Dataset '{args.dataset}' not found in {resolution}x{resolution} registry"
                    )
                    continue

            for _, ds_meta in datasets_to_check:
                try:
                    run_logger.info(f"Attempting health check for dataset '{ds_meta.name}'")
                    health_check_single_dataset(ds_meta, orchestrator, resolution=resolution)
                except Exception as e:
                    run_logger.warning(
                        f"Skipping dataset '{ds_meta.name}' due to error, but continuing."
                    )
                    run_logger.exception(e)

            run_logger.info(f"Health check completed for resolution {resolution}x{resolution}.")


# ENTRY POINT
if __name__ == "__main__":
    fetch_all_datasets_health_check()
