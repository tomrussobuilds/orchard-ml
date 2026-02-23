"""
Scheduler Stepping Utility.

Provides a shared helper for stepping learning rate schedulers,
handling the ReduceLROnPlateau special case.
"""

from __future__ import annotations

from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau


def step_scheduler(scheduler: LRScheduler | ReduceLROnPlateau | None, val_loss: float) -> None:
    """
    Step the learning rate scheduler.

    ReduceLROnPlateau requires the monitored metric (val_loss);
    all other schedulers use a plain step().

    Args:
        scheduler: Learning rate scheduler instance, or None (no-op).
        val_loss: Current epoch's validation loss.
    """
    if scheduler is None:
        return

    if isinstance(scheduler, ReduceLROnPlateau):
        scheduler.step(val_loss)
    else:
        scheduler.step()
