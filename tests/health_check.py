"""
Health Check and Integrity Module (Multi-Resolution with Visualization)
Performs integrity checks across MedMNIST datasets, supports 28x28 and 224x224 resolutions.

Usage:
    python -m tests.health_check
    python -m tests.health_check --dataset bloodmnist --resolution 28
"""

from __future__ import annotations

import argparse
import logging

from orchard.core import Config, RootOrchestrator
from orchard.core.metadata import DatasetRegistryWrapper
from orchard.core.paths import HEALTHCHECK_LOGGER_NAME
from orchard.data_handler.data_explorer import show_samples_for_dataset
from orchard.data_handler.fetcher import load_dataset_health_check
from orchard.data_handler.loader import create_temp_loader

# Logging Setup
logger = logging.getLogger(HEALTHCHECK_LOGGER_NAME)
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


def _build_health_check_config(dataset: str, resolution: int) -> Config:
    """Build a minimal Config for health check (no YAML needed)."""
    arch = "mini_cnn" if resolution == 28 else "efficientnet_b0"
    return Config(
        dataset={"name": dataset, "resolution": resolution, "max_samples": 100},
        architecture={"name": arch, "pretrained": False},
        training={
            "epochs": 1,
            "batch_size": 32,
            "learning_rate": 0.001,
            "use_amp": False,
            "mixup_alpha": 0.0,
            "mixup_epochs": 0,
        },
        hardware={"device": "cpu", "project_name": "health-check", "reproducible": True},
        telemetry={"data_dir": "./dataset", "output_dir": "./outputs"},
    )


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
            dataset_name=ds_meta.name,
            run_paths=orchestrator.paths,
            resolution=resolution,
        )

        run_logger.info(f"Sample images saved for dataset: {ds_meta.display_name}")
        run_logger.info(f"Health check PASSED for dataset: {ds_meta.display_name}")

    except (FileNotFoundError, ValueError, RuntimeError) as e:
        run_logger.error(
            f"Health check FAILED for dataset: {ds_meta.display_name} ({ds_meta.name})"
        )
        run_logger.exception(e)


def _filter_datasets(registry_items, dataset_name, resolution, run_logger):
    """Filter dataset registry by name, returning None if the requested dataset is not found."""
    if not dataset_name:
        return list(registry_items)

    filtered = [(name, meta) for name, meta in registry_items if name == dataset_name]
    if not filtered:
        run_logger.error(
            f"Dataset '{dataset_name}' not found in {resolution}x{resolution} registry"
        )
    return filtered or None


def fetch_all_datasets_health_check() -> None:
    """
    Iterates over the MedMNIST dataset registry and performs a health check for each dataset.
    """
    args = parse_health_check_args()
    resolutions = [args.resolution] if args.resolution is not None else [28, 224]

    for resolution in resolutions:
        cfg = _build_health_check_config(
            dataset=args.dataset or "bloodmnist",
            resolution=resolution,
        )

        with RootOrchestrator(cfg) as orchestrator:
            run_logger = orchestrator.run_logger
            run_logger.info(f"Checking datasets with resolution: {resolution}x{resolution}")

            dataset_wrapper = DatasetRegistryWrapper(resolution=resolution)
            datasets_to_check = _filter_datasets(
                dataset_wrapper.registry.items(), args.dataset, resolution, run_logger
            )
            if datasets_to_check is None:
                continue

            for _, ds_meta in datasets_to_check:
                try:
                    run_logger.info(f"Attempting health check for dataset '{ds_meta.name}'")
                    health_check_single_dataset(ds_meta, orchestrator, resolution=resolution)
                except (FileNotFoundError, ValueError, RuntimeError) as e:
                    run_logger.warning(
                        f"Skipping dataset '{ds_meta.name}' due to error, but continuing."
                    )
                    run_logger.exception(e)

            run_logger.info(f"Health check completed for resolution {resolution}x{resolution}.")


# ENTRY POINT
if __name__ == "__main__":
    fetch_all_datasets_health_check()
