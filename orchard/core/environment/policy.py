"""
Execution & Optimization Policies.

Defines decision-making logic for runtime strategy selection based on
hardware availability and configuration constraints.

This module contains policy functions that determine optimal execution
strategies (e.g., TTA mode complexity) by analyzing available resources
and user configuration. Policies ensure the framework adapts intelligently
to heterogeneous deployment environments (CPU, CUDA, MPS).

Key Policies:
    * TTA Mode Selection: Balances augmentation ensemble size with hardware
      acceleration to prevent CPU bottlenecks while maximizing GPU throughput
"""


def determine_tta_mode(use_tta: bool, device_type: str, tta_mode: str = "full") -> str:
    """
    Reports the active TTA ensemble policy.

    The ensemble complexity is driven by the ``tta_mode`` config field,
    not by hardware.  This guarantees identical predictions on CPU, CUDA
    and MPS for the same config, preserving cross-platform determinism.

    Args:
        use_tta: Whether Test-Time Augmentation is enabled.
        device_type: The type of active device ('cpu', 'cuda', 'mps').
        tta_mode: Config-driven ensemble complexity ('full' or 'light').

    Returns:
        Descriptive string of the TTA operation mode.
    """
    if not use_tta:
        return "DISABLED"

    mode_label = tta_mode.upper()
    return f"{mode_label} ({device_type.upper()})"
