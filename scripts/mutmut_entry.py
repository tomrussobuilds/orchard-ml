#!/usr/bin/env python3
"""
Patched mutmut entry point that skips mutations on logging and warning calls.

This avoids scattering ``# pragma: no mutate`` across every logger/warnings
line.  The patch extends ``_skip_node_and_children`` to recognise method calls
like ``logger.info(...)``, ``log.debug(...)``, ``warnings.warn(...)`` etc. and
skip the entire AST node (call + all children).

Usage (drop-in replacement for ``python -m mutmut``):

    python scripts/mutmut_entry.py run 'orchard.cli_app*'
"""

from __future__ import annotations

import runpy

import libcst as cst
import mutmut.file_mutation as fm

# ---------------------------------------------------------------------------
# Logging method names whose calls (and all their arguments) should never be
# mutated.  We match ``<anything>.<method>(...)`` where method is in this set.
# ---------------------------------------------------------------------------
_SKIP_METHODS: frozenset[str] = frozenset(
    {
        "debug",
        "info",
        "warn",  # warnings.warn
    }
)

_original_skip = fm.MutationVisitor._skip_node_and_children


def _patched_skip(self: fm.MutationVisitor, node: cst.CSTNode) -> bool:
    if _original_skip(self, node):
        return True

    # Skip  obj.method(...)  where method is a logging/warning helper.
    if (
        isinstance(node, cst.Call)
        and isinstance(node.func, cst.Attribute)
        and node.func.attr.value in _SKIP_METHODS
    ):
        return True

    return False


fm.MutationVisitor._skip_node_and_children = _patched_skip

# Forward to mutmut CLI (__main__) ------------------------------------------
runpy.run_module("mutmut", run_name="__main__", alter_sys=True)
