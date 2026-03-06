#!/usr/bin/env python3
"""
Mutation registry guard — prevents score regressions and enforces freshness.

Modes
-----
--ratchet    Compare the working-tree registry against the last committed version
             (git HEAD).  Fails if any module's score dropped.  Intended for
             pre-commit: only meaningful when mutmut-registry.yaml is staged.

--freshness  For every .py file under orchard/ that was modified after its
             ``last_run`` timestamp in the registry, report it as stale.
             Files with no entry at all are also flagged.  Intended as a
             release gate: you only need to re-run mutmut on modules you
             actually changed.

Both flags can be combined.

Usage::

    python scripts/check_mutmut_registry.py --ratchet
    python scripts/check_mutmut_registry.py --freshness
    python scripts/check_mutmut_registry.py --ratchet --freshness
"""

from __future__ import annotations

import argparse
import subprocess  # nosec B404
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    sys.exit("PyYAML is required: pip install pyyaml")

ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = ROOT / "mutmut-registry.yaml"
ORCHARD_DIR = ROOT / "orchard"


def _load_registry(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _load_head_registry() -> dict[str, Any]:
    """Load mutmut-registry.yaml from git HEAD."""
    result = subprocess.run(  # nosec B603 B607
        ["git", "show", "HEAD:mutmut-registry.yaml"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    if result.returncode != 0:
        return {}
    return yaml.safe_load(result.stdout) or {}


def _git_last_modified(path: str) -> datetime | None:
    """Return the author-date of the last commit that touched *path*."""
    result = subprocess.run(  # nosec B603 B607
        ["git", "log", "-1", "--format=%aI", "--", path],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    line = result.stdout.strip()
    if not line:
        return None
    return datetime.fromisoformat(line)


def check_ratchet() -> list[str]:
    """Return list of error messages for modules whose score dropped."""
    current = _load_registry(REGISTRY_PATH)
    baseline = _load_head_registry()
    errors: list[str] = []

    for module, base_entry in baseline.items():
        if module not in current:
            continue  # removed module, OK
        cur_score = current[module].get("score", 0.0)
        base_score = base_entry.get("score", 0.0)
        if cur_score < base_score:
            errors.append(
                f"  {module}: {base_score:.1f}% -> {cur_score:.1f}% (dropped {base_score - cur_score:.1f}pp)"
            )

    return errors


def check_freshness() -> list[str]:
    """Return list of modules whose registry entry is stale or missing."""
    current = _load_registry(REGISTRY_PATH)
    stale: list[str] = []

    for py_file in sorted(ORCHARD_DIR.rglob("*.py")):
        if py_file.name == "__init__.py":
            continue
        if "__pycache__" in py_file.parts:
            continue

        key = str(py_file.relative_to(ROOT))

        if key not in current:
            stale.append(f"  {key} (no entry)")
            continue

        last_run_str = current[key].get("last_run", "")
        if not last_run_str:
            stale.append(f"  {key} (no last_run)")
            continue

        last_run = datetime.fromisoformat(last_run_str).replace(tzinfo=timezone.utc)

        file_modified = _git_last_modified(key)
        if file_modified is None:
            continue  # untracked or new file not yet committed, skip

        if file_modified > last_run:
            stale.append(
                f"  {key} (modified {file_modified:%Y-%m-%d}, last_run {last_run:%Y-%m-%d})"
            )

    return stale


def main() -> None:
    """CLI entry point for mutation registry checks."""
    parser = argparse.ArgumentParser(description="Mutation registry guard.")
    parser.add_argument(
        "--ratchet",
        action="store_true",
        help="Fail if any module score dropped vs HEAD",
    )
    parser.add_argument(
        "--freshness",
        action="store_true",
        help="Fail if any modified module has a stale or missing registry entry",
    )
    args = parser.parse_args()

    if not args.ratchet and not args.freshness:
        parser.print_help()
        sys.exit(1)

    failed = False

    if args.ratchet:
        errors = check_ratchet()
        if errors:
            print("Mutation score regression detected:")
            print("\n".join(errors))
            failed = True
        else:
            print("Ratchet check passed: no score regressions.")

    if args.freshness:
        stale = check_freshness()
        if stale:
            print("Stale or missing mutation entries (re-run mutmut on these):")
            print("\n".join(stale))
            failed = True
        else:
            print("Freshness check passed: all modules up to date.")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
