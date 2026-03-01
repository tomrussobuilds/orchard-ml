"""
Orchard ML Exception Hierarchy.

OrchardError (base, Exception)
├── OrchardConfigError(OrchardError, ValueError)   ← config validation
├── OrchardDatasetError(OrchardError)              ← data I/O and fetching
└── OrchardExportError(OrchardError)               ← model export failures

OrchardConfigError multi-inherits from ValueError to preserve
backward compatibility with existing ``except ValueError`` blocks.
"""


class OrchardError(Exception):
    """Base exception for all Orchard ML errors."""


class OrchardConfigError(OrchardError, ValueError):
    """Configuration validation error (backward-compatible with ValueError)."""


class OrchardDatasetError(OrchardError):
    """Dataset loading, fetching, or validation error."""


class OrchardExportError(OrchardError):
    """Model export (ONNX) or checkpoint loading error."""
