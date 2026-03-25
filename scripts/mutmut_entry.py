#!/usr/bin/env python3
"""
Patched mutmut entry point that skips mutations on logging and warning calls.

This avoids scattering ``# pragma: no mutate`` across every logger/warnings
line.  Two suppression levels are provided:

1. **Full skip** (``_SKIP_METHODS``): the entire ``Call`` node is excluded —
   call, arguments, and string content.  Used for low-severity log calls
   (``debug``, ``info``) and ``warnings.warn`` whose removal has no
   observable effect in tests.

2. **String-only skip** (``_SKIP_LOG_STRINGS_METHODS``): only string literal
   children (``SimpleString``, ``ConcatenatedString``, ``FormattedString``)
   inside the call are excluded.  The call itself and non-string arguments
   remain mutable so that tests still verify the log call happens with the
   correct arguments.  Used for ``warning``, ``error``, and ``getLogger``.

Usage (drop-in replacement for ``python -m mutmut``):

    python scripts/mutmut_entry.py run 'orchard.cli_app*'
"""

from __future__ import annotations

import runpy

import libcst as cst
import mutmut.file_mutation as fm

# ---------------------------------------------------------------------------
# Full-skip: the entire Call node + children are excluded from mutation.
# ---------------------------------------------------------------------------
_SKIP_METHODS: frozenset[str] = frozenset(
    {
        "debug",
        "info",
        "add_format",  # xlsxwriter cosmetic formatting dicts
    }
)

# ---------------------------------------------------------------------------
# String-only skip: only string-literal children of these calls are excluded.
# The call itself and non-string args remain mutable.
# ---------------------------------------------------------------------------
_SKIP_LOG_STRINGS_METHODS: frozenset[str] = frozenset(
    {
        "warning",
        "error",
        "warn",  # warnings.warn
        "getLogger",
    }
)

_original_skip = fm.MutationVisitor._skip_node_and_children

_STRING_TYPES = (cst.SimpleString, cst.ConcatenatedString, cst.FormattedString)

# ---------------------------------------------------------------------------
# Matplotlib receiver skip: calls on plt/ax/fig/disp objects are cosmetic
# (colors, fonts, layout) and not observable via unit tests.
# ---------------------------------------------------------------------------
_MATPLOTLIB_RECEIVERS: frozenset[str] = frozenset(
    {
        "plt",
        "plt_obj",
        "ax",
        "ax1",
        "ax2",
        "fig",
        "disp",
    }
)

# ---------------------------------------------------------------------------
# Cosmetic receiver skip: calls on these receivers are display-only.
# ---------------------------------------------------------------------------
_COSMETIC_RECEIVERS: frozenset[str] = frozenset(
    _MATPLOTLIB_RECEIVERS
    | {
        "Reporter",  # Reporter.log_phase_header(...)
    }
)

# ---------------------------------------------------------------------------
# Cosmetic function skip: bare function calls whose args are purely visual.
# Unlike receivers, these are called as ``func(...)`` not ``obj.method(...)``.
# ---------------------------------------------------------------------------
_COSMETIC_FUNCTIONS: frozenset[str] = frozenset(
    {
        "_draw_box",  # forwards color/lw/linestyle to Rectangle
        "_draw_label",  # forwards text/color/position to ax.text
        "Rectangle",  # matplotlib.patches.Rectangle constructor
        "show_detections",  # forwards model/loader/ctx to visualization
        "show_predictions",  # classification equivalent
    }
)

# ---------------------------------------------------------------------------
# LogStyle attribute skip: assignments like ``I = LogStyle.INDENT`` are
# cosmetic aliases.  We skip the ``LogStyle.X`` Attribute node itself.
# ---------------------------------------------------------------------------
_LOGSTYLE_ATTRS: frozenset[str] = frozenset(
    {
        "INDENT",
        "ARROW",
        "BULLET",
        "SUCCESS",
        "WARNING",
        "FAILURE",
        "DOUBLE",
        "LIGHT",
        "HEADER_WIDTH",
    }
)


def _get_receiver_name(node: cst.Call) -> str | None:
    """Extract the receiver name from ``receiver.method(...)`` calls."""
    if not isinstance(node.func, cst.Attribute):
        return None
    value = node.func.value
    # Simple name: plt.savefig(...)
    if isinstance(value, cst.Name):
        return value.value
    # Chained attribute: plt.style.context(...)
    if isinstance(value, cst.Attribute) and isinstance(value.value, cst.Name):
        return value.value.value
    return None


def _is_logstyle_assign(node: cst.Assign) -> bool:
    """Return True if *node* is ``X = LogStyle.ATTR`` with ATTR in _LOGSTYLE_ATTRS."""
    val = node.value
    return (
        isinstance(val, cst.Attribute)
        and isinstance(val.value, cst.Name)
        and val.value.value == "LogStyle"
        and val.attr.value in _LOGSTYLE_ATTRS
    )


def _patched_skip(self: fm.MutationVisitor, node: cst.CSTNode) -> bool:
    if _original_skip(self, node):
        return True

    # Full skip:  obj.method(...)  where method is a low-severity log helper.
    if (
        isinstance(node, cst.Call)
        and isinstance(node.func, cst.Attribute)
        and node.func.attr.value in _SKIP_METHODS
    ):
        return True

    # Cosmetic receiver skip: plt.savefig(...), ax.set_title(...),
    # Reporter.log_phase_header(...), etc.
    if isinstance(node, cst.Call) and _get_receiver_name(node) in _COSMETIC_RECEIVERS:
        return True

    # Cosmetic function skip: _draw_box(...), _draw_label(...), etc.
    if (
        isinstance(node, cst.Call)
        and isinstance(node.func, cst.Name)
        and node.func.value in _COSMETIC_FUNCTIONS
    ):
        return True

    # LogStyle attribute skip: ``LogStyle.INDENT`` and similar constants,
    # plus assignments like ``I = LogStyle.INDENT`` → ``I = None``.
    if (
        isinstance(node, cst.Attribute)
        and isinstance(node.value, cst.Name)
        and node.value.value == "LogStyle"
        and node.attr.value in _LOGSTYLE_ATTRS
    ):
        return True
    if isinstance(node, cst.Assign) and _is_logstyle_assign(node):
        return True

    # String-only skip: inside a warning/error/getLogger call, skip string
    # literals and Args whose value is a string literal (catches the
    # ``string → None`` Arg-level replacement mutants too).
    if getattr(self, "_log_string_depth", 0) > 0:
        if isinstance(node, _STRING_TYPES):
            return True
        if isinstance(node, cst.Arg) and isinstance(node.value, _STRING_TYPES):
            return True

    return False


# ---------------------------------------------------------------------------
# State tracking: increment/decrement a depth counter when entering/leaving
# a Call whose method name is in _SKIP_LOG_STRINGS_METHODS.
# ---------------------------------------------------------------------------
_original_on_visit = fm.MutationVisitor.on_visit
_original_on_leave = fm.MutationVisitor.on_leave


def _patched_on_visit(self: fm.MutationVisitor, node: cst.CSTNode) -> bool:
    if (
        isinstance(node, cst.Call)
        and isinstance(node.func, cst.Attribute)
        and node.func.attr.value in _SKIP_LOG_STRINGS_METHODS
    ):
        self._log_string_depth = getattr(self, "_log_string_depth", 0) + 1
    result: bool = _original_on_visit(self, node)
    return result


def _patched_on_leave(self: fm.MutationVisitor, original_node: cst.CSTNode) -> None:
    _original_on_leave(self, original_node)
    if (
        isinstance(original_node, cst.Call)
        and isinstance(original_node.func, cst.Attribute)
        and original_node.func.attr.value in _SKIP_LOG_STRINGS_METHODS
    ):
        self._log_string_depth = getattr(self, "_log_string_depth", 1) - 1


# ---------------------------------------------------------------------------
# Post-filter: ``operator_arg_removal`` generates Call-level mutations that
# replace each Arg.value with ``None`` or remove an Arg entirely.  When the
# original Arg holds a string literal inside a _SKIP_LOG_STRINGS_METHODS
# call, these mutations are cosmetic (the log message changes) and should be
# discarded.
# ---------------------------------------------------------------------------
_original_create_mutations = fm.MutationVisitor._create_mutations


def _patched_create_mutations(self: fm.MutationVisitor, node: cst.CSTNode) -> None:
    if not (getattr(self, "_log_string_depth", 0) > 0 and isinstance(node, cst.Call)):
        _original_create_mutations(self, node)
        return

    # Snapshot length, create mutations, then drop the unwanted ones.
    before = len(self.mutations)
    _original_create_mutations(self, node)

    # Identify which original arg positions hold string literals.
    string_arg_indices = {
        i for i, arg in enumerate(node.args) if isinstance(arg.value, _STRING_TYPES)
    }
    if not string_arg_indices:
        return  # nothing to filter

    # Drop mutations that *only* differ in string-arg positions.
    filtered: list[fm.Mutation] = []
    for m in self.mutations[before:]:
        if not isinstance(m.mutated_node, cst.Call):
            filtered.append(m)
            continue

        drop = False
        orig_args = node.args
        mut_args = m.mutated_node.args

        # Case 1: arg replaced with None (same arg count)
        if len(orig_args) == len(mut_args):
            changed = [i for i in range(len(orig_args)) if orig_args[i] is not mut_args[i]]
            if changed and all(i in string_arg_indices for i in changed):
                drop = True

        # Case 2: arg removed (one fewer arg)
        if len(mut_args) == len(orig_args) - 1:
            for removed_idx in string_arg_indices:
                remaining = [*orig_args[:removed_idx], *orig_args[removed_idx + 1 :]]
                if len(remaining) == len(mut_args):
                    if all(
                        remaining[j].value.deep_equals(mut_args[j].value)
                        for j in range(len(mut_args))
                    ):
                        drop = True
                        break

        if not drop:
            filtered.append(m)

    self.mutations[before:] = filtered


fm.MutationVisitor._skip_node_and_children = _patched_skip
fm.MutationVisitor.on_visit = _patched_on_visit
fm.MutationVisitor.on_leave = _patched_on_leave
fm.MutationVisitor._create_mutations = _patched_create_mutations

# Forward to mutmut CLI (__main__) ------------------------------------------
runpy.run_module("mutmut", run_name="__main__", alter_sys=True)
