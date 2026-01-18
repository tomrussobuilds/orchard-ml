"""
Hyperparameter Optimization Entry Point.

Orchestrates Optuna-driven hyperparameter search for the classification
pipeline. Integrates seamlessly with the existing Pydantic configuration engine
and RootOrchestrator lifecycle management.

Workflow:
    1. Parse CLI arguments (base + Optuna-specific)
    2. Build unified Config (including OptunaConfig)
    3. Initialize RootOrchestrator (device, paths, logging)
    4. Execute Optuna optimization study
    5. Generate visualizations and export best configuration
    6. Save study results for reproducibility

Usage:
    # Quick search
    python optimize.py --config recipes/config_28x28_resnet_18_adapted.yaml
    
    # Custom parameters
    python optimize.py --dataset pathmnist --n_trials 50 --epochs 12
    
    # Resume interrupted study
    python optimize.py --config recipes/config_resnet_18_adapted.yaml --load_if_exists true
"""

# =========================================================================== #
#                           INTERNAL IMPORTS                                  #
# =========================================================================== #
from orchard.core import RootOrchestrator, Config, parse_args
from orchard.optimization import run_optimization

# =========================================================================== #
#                           MAIN EXECUTION                                    #
# =========================================================================== #

def main():
    """
    Main orchestrator for hyperparameter optimization.
    
    Coordinates the complete optimization lifecycle from configuration parsing
    to final result reporting. Utilizes RootOrchestrator context manager for
    resource safety and automatic cleanup.
    
    Workflow:
        1. Parse CLI arguments (includes Optuna-specific flags)
        2. Build unified Config with OptunaConfig validation
        3. Initialize orchestrator (device, filesystem, logging)
        4. Execute Optuna study with trial pruning
        5. Generate importance plots and export best config
        6. Report final statistics and next steps
    
    Raises:
        KeyboardInterrupt: User interrupted optimization (graceful cleanup)
        Exception: Any fatal error during optimization (logged and re-raised)
    """
    # Parse CLI arguments (supports both YAML and direct flags)
    args = parse_args()
    
    # Build configurations (triggers Pydantic validation)
    config = Config.from_args(args)
    
    # Use orchestrator context manager for resource safety
    # Guarantees cleanup even if optimization crashes
    with RootOrchestrator(config) as orchestrator:

        device = orchestrator.get_device()
        paths = orchestrator.paths
        logger = orchestrator.run_logger
        
        try:
            # --- Execute Optimization ---
            study = run_optimization(
                cfg=config,
                device=device,
                paths=paths
            )
            
            # --- Final Summary ---
            logger.info(
                f"\n{'#' * 80}\n"
                f"{'OPTIMIZATION COMPLETE':^80}\n"
                f"{'#' * 80}\n"
                f"  Best {config.optuna.metric_name}: {study.best_value:.4f}\n"
                f"  Best Trial: {study.best_trial.number}\n"
                f"  Completed Trials: {len([t for t in study.trials if t.state.name == 'COMPLETE'])}\n"
                f"  Pruned Trials: {len([t for t in study.trials if t.state.name == 'PRUNED'])}\n"
                f"  Results: {paths.root}\n"
                f"{'#' * 80}"
            )
        
        except KeyboardInterrupt:
            logger.warning("\n[!] Interrupted by user. Cleaning up and exiting...")
        except Exception as e:
            logger.error(f"\n[!] Pipeline crashed during execution: {e}", exc_info=True)
            raise e
        
        finally:
            # The Context Manager (__exit__) handles orchestrator.cleanup() automatically.
            # This releases the infrastructure lock and closes logging handlers.
            if 'paths' in locals() and paths:
                logger.info(f"Pipeline Shutdown completed. Run directory: {paths.root}")


# =========================================================================== #
#                           ENTRY POINT                                       #
# =========================================================================== #

if __name__ == "__main__":
    main()