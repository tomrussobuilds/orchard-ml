"""
Scheduler Stepping Utility.

Provides a shared helper for stepping learning rate schedulers,
handling the ReduceLROnPlateau special case.
"""

from __future__ import annotations

from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau


def step_scheduler(scheduler: LRScheduler | ReduceLROnPlateau | None, monitor_value: float) -> None:
    """
    Step the learning rate scheduler.

    ReduceLROnPlateau requires the monitored metric value (e.g. accuracy,
    auc, or loss — whichever ``training.monitor_metric`` specifies);
    all other schedulers use a plain step().

    Args:
        scheduler: Learning rate scheduler instance, or None (no-op).
        monitor_value: Current epoch's value of the monitored metric.
    """
    if scheduler is None:
        return

    if isinstance(scheduler, ReduceLROnPlateau):
        scheduler.step(monitor_value)
    else:
        scheduler.step()
