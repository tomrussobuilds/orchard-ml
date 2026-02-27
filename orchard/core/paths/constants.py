"""
Project-wide Constants, Metric Keys, and Logging Style.

Single source of truth for static values used across all layers of the
codebase: paths, environment, config, and logger.  Lives in the lowest-level
package (``paths``) so that every other module can safely import from here
without circular-dependency risks.

Module Attributes:
    SUPPORTED_RESOLUTIONS: Valid image resolutions across architectures.
    METRIC_*: Canonical metric key strings.
    LOGGER_NAME: Global logger identity for log synchronization.
    HEALTHCHECK_LOGGER_NAME: Logger identity for dataset validation.
    LogStyle: Unified logging style constants.
"""

from __future__ import annotations

from typing import Final

# GLOBAL CONSTANTS
# Supported image resolutions across all model architectures
SUPPORTED_RESOLUTIONS: Final[frozenset[int]] = frozenset({28, 32, 64, 128, 224})

# Canonical metric key strings used across training, evaluation, and optimization
METRIC_ACCURACY: Final[str] = "accuracy"
METRIC_AUC: Final[str] = "auc"
METRIC_LOSS: Final[str] = "loss"
METRIC_F1: Final[str] = "f1"

# Global logger identity used by all modules to ensure log synchronization
LOGGER_NAME: Final[str] = "OrchardML"

# Health check logger identity for dataset validation utilities
HEALTHCHECK_LOGGER_NAME: Final[str] = "healthcheck"


class LogStyle:
    """
    Unified logging style constants for consistent visual hierarchy.

    Provides separators, symbols, indentation, and ANSI color codes used
    by all logging modules.  Placed here (in ``paths.constants``) rather
    than in ``logger.styles`` so that low-level packages (``environment``,
    ``config``) can reference the constants without triggering circular
    imports.
    """

    # Header centering width (matches separator length)
    HEADER_WIDTH = 80

    # Level 1: Session headers (80 chars)
    HEAVY = "━" * HEADER_WIDTH

    # Level 2: Major sections (80 chars)
    DOUBLE = "═" * HEADER_WIDTH

    # Level 3: Subsections / Separators (80 chars)
    LIGHT = "─" * HEADER_WIDTH

    # Symbols
    ARROW = "»"
    BULLET = "•"
    WARNING = "⚠"
    SUCCESS = "✓"

    # Indentation
    INDENT = "  "
    DOUBLE_INDENT = "    "

    # ANSI Colors (applied by ColorFormatter to console output only)
    RESET = "\033[0m"
    DIM = "\033[2m"
    BOLD = "\033[1m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    CYAN = "\033[36m"
    MAGENTA = "\033[35m"
