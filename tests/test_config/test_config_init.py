"""
Test Suite for config package lazy-loading init.

Validates PEP 562 __getattr__ lazy imports, __dir__ support,
and error handling for unknown attributes.
"""

from __future__ import annotations

import pytest

from orchard.core.config import _LAZY_IMPORTS, __all__


@pytest.mark.unit
@pytest.mark.parametrize("name", __all__)
def test_lazy_import_resolves(name: str) -> None:
    """Every name in __all__ is importable via lazy loading."""
    from orchard.core import config

    attr = getattr(config, name)
    assert attr is not None


@pytest.mark.unit
def test_lazy_import_is_cached() -> None:
    """Second access hits globals() cache, not __getattr__."""
    from orchard.core import config

    first = getattr(config, "Config")
    second = getattr(config, "Config")
    assert first is second


@pytest.mark.unit
def test_invalid_attr_raises() -> None:
    """Unknown attribute raises AttributeError with module name."""
    from orchard.core import config

    with pytest.raises(AttributeError, match="has no attribute"):
        getattr(config, "NonExistentConfig")


@pytest.mark.unit
def test_dir_contains_all() -> None:
    """dir(config) includes every public name."""
    from orchard.core import config

    public = dir(config)
    for name in __all__:
        assert name in public


@pytest.mark.unit
def test_dir_is_sorted() -> None:
    """dir(config) returns a sorted list."""
    from orchard.core import config

    public = dir(config)
    assert public == sorted(public)


@pytest.mark.unit
def test_all_matches_lazy_imports() -> None:
    """__all__ and _LAZY_IMPORTS have the same keys."""
    assert set(__all__) == set(_LAZY_IMPORTS.keys())
