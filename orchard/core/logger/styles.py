"""
Logging style constants for consistent visual hierarchy.

Provides unified formatting symbols and separators used across all logging modules.
"""

from __future__ import annotations

import logging


class LogStyle:
    """Unified logging style constants for consistent visual hierarchy."""

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

    @staticmethod
    def log_phase_header(
        log: logging.Logger,
        title: str,
        style: str | None = None,
    ) -> None:
        """
        Log a centered phase header with separator lines.

        Args:
            log: Logger instance to write to.
            title: Header text (will be uppercased and centered).
            style: Separator string (defaults to ``LogStyle.HEAVY``).
        """
        sep = style if style is not None else LogStyle.HEAVY
        log.info("")
        log.info(sep)
        log.info(f"{title:^{LogStyle.HEADER_WIDTH}}")
        log.info(sep)
