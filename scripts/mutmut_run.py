#!/usr/bin/env python3
"""
Run mutmut on specific modules and update the mutation registry.

Usage:
    # Single file
    python scripts/mutmut_run.py orchard/cli_app.py

    # Whole sub-package
    python scripts/mutmut_run.py orchard/core/config/

    # Multiple targets
    python scripts/mutmut_run.py orchard/cli_app.py orchard/exceptions.py

    # Report only (no mutmut run, just re-parse existing .meta files)
    python scripts/mutmut_run.py --report

    # Report for a specific module
    python scripts/mutmut_run.py --report orchard/cli_app.py

    # Batch: run each .py file one by one (cleans cache, updates registry after each)
    python scripts/mutmut_run.py --batch orchard/trainer/

    # Batch the whole project
    python scripts/mutmut_run.py --batch orchard/
"""

from __future__ import annotations

import argparse
import json
import subprocess  # nosec B404 — runs mutmut CLI, no untrusted input
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    sys.exit("PyYAML is required: pip install pyyaml")

ROOT = Path(__file__).resolve().parent.parent
ENTRY_SCRIPT = str(ROOT / "scripts" / "mutmut_entry.py")
MUTANTS_DIR = ROOT / "mutants"
REGISTRY_PATH = ROOT / "mutmut-registry.yaml"


def _source_files(target: str) -> list[Path]:
    """Resolve a target (file or directory) to a list of .py source files."""
    p = ROOT / target
    if p.is_file() and p.suffix == ".py":
        return [p]
    if p.is_dir():
        return sorted(f for f in p.rglob("*.py") if f.name != "__pycache__")
    sys.exit(f"Target not found: {target}")
    return []  # unreachable — satisfies static analysis


def _to_mutmut_glob(source: Path) -> str:
    """Convert a source path to a mutmut glob pattern.

    Example: orchard/core/config/dataset_config.py -> orchard.core.config.dataset_config*
    """
    rel = source.relative_to(ROOT)
    dotted = str(rel).replace("/", ".").removesuffix(".py")
    # __init__.py maps to the package module, not package.__init__
    dotted = dotted.removesuffix(".__init__")
    return f"{dotted}*"


def _parse_meta(meta_path: Path) -> dict[str, int]:
    """Parse a .meta file and return counts: total, killed, survived, not_checked."""
    if not meta_path.exists():
        return {"total": 0, "killed": 0, "survived": 0, "not_checked": 0}

    with open(meta_path) as f:
        data = json.load(f)

    exit_codes = data.get("exit_code_by_key", {})
    total = len(exit_codes)
    killed = sum(1 for v in exit_codes.values() if v is not None and v != 0)
    survived = sum(1 for v in exit_codes.values() if v == 0)
    not_checked = sum(1 for v in exit_codes.values() if v is None)

    return {
        "total": total,
        "killed": killed,
        "survived": survived,
        "not_checked": not_checked,
    }


def _meta_path_for(source: Path) -> Path:
    """Get the .meta path in mutants/ corresponding to a source file."""
    rel = source.relative_to(ROOT)
    return MUTANTS_DIR / f"{rel}.meta"


def _update_registry(sources: list[Path]) -> dict[str, Any]:
    """Parse .meta files and update the registry YAML. Returns updated entries."""
    registry: dict[str, Any] = {}
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH) as f:
            registry = yaml.safe_load(f) or {}

    updated: dict[str, Any] = {}
    for src in sources:
        meta = _meta_path_for(src)
        counts = _parse_meta(meta)

        key = str(src.relative_to(ROOT))
        score = round(counts["killed"] / counts["total"] * 100, 1) if counts["total"] else 100.0

        entry = {
            "last_run": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "total": counts["total"],
            "killed": counts["killed"],
            "survived": counts["survived"],
            "not_checked": counts["not_checked"],
            "score": score,
        }
        registry[key] = entry
        updated[key] = entry

    # Sort registry by key for stable output
    registry = dict(sorted(registry.items()))

    with open(REGISTRY_PATH, "w") as f:
        yaml.dump(registry, f, default_flow_style=False, sort_keys=False)

    return updated


def _print_table(entries: dict[str, Any]) -> None:
    """Print a summary table of mutation results."""
    if not entries:
        print("No mutation results found.")
        return

    header = f"{'Module':<55} {'Total':>5} {'Kill':>5} {'Surv':>5} {'N/C':>5} {'Score':>7}"
    print(header)
    print("-" * len(header))

    for key, e in sorted(entries.items()):
        score_str = f"{e['score']:.1f}%"
        print(
            f"{key:<55} {e['total']:>5} {e['killed']:>5} "
            f"{e['survived']:>5} {e['not_checked']:>5} {score_str:>7}"
        )

    # Summary
    totals = {
        k: sum(e[k] for e in entries.values())
        for k in ("total", "killed", "survived", "not_checked")
    }
    overall = round(totals["killed"] / totals["total"] * 100, 1) if totals["total"] else 0.0
    print("-" * len(header))
    print(
        f"{'TOTAL':<55} {totals['total']:>5} {totals['killed']:>5} "
        f"{totals['survived']:>5} {totals['not_checked']:>5} {overall:.1f}%"
    )


def _is_fresh(source: Path, registry: dict[str, Any]) -> bool:
    """Return True if the registry entry for *source* is newer than its last git commit."""
    key = str(source.relative_to(ROOT))
    entry = registry.get(key)
    if not entry or not entry.get("last_run"):
        return False

    last_run = datetime.fromisoformat(entry["last_run"].replace("Z", "+00:00")).replace(
        tzinfo=timezone.utc
    )

    result = subprocess.run(  # nosec B603 B607
        ["git", "log", "-1", "--format=%aI", "--", key],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    line = result.stdout.strip()
    if not line:
        return False

    file_modified = datetime.fromisoformat(line.replace("Z", "+00:00"))
    if file_modified.tzinfo is None:
        file_modified = file_modified.replace(tzinfo=timezone.utc)

    return last_run > file_modified


def _clean_cache(source: Path, *, backup_meta: bool = False) -> None:
    """Remove cached trampoline and meta for a source file.

    When *backup_meta* is True the existing ``.meta`` is renamed to
    ``.meta.bak`` instead of deleted so it can be restored if the new
    run produces incomplete results.
    """
    rel = source.relative_to(ROOT)
    trampoline = MUTANTS_DIR / rel
    meta = MUTANTS_DIR / f"{rel}.meta"
    if trampoline.exists():
        trampoline.unlink()
    if meta.exists():
        if backup_meta:
            meta.rename(meta.with_suffix(".meta.bak"))
        else:
            meta.unlink()


def run_mutmut(targets: list[str]) -> None:
    """Run mutmut on the given targets."""
    sources: list[Path] = []
    globs: list[str] = []

    for target in targets:
        files = _source_files(target)
        sources.extend(files)
        globs.extend(_to_mutmut_glob(f) for f in files)

    if not globs:
        sys.exit("No .py files found for the given targets.")

    print(f"Running mutmut on {len(sources)} file(s)...")
    cmd = [sys.executable, ENTRY_SCRIPT, "run", *globs]
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT)  # nosec B603

    print("\nUpdating registry...")
    updated = _update_registry(sources)
    _print_table(updated)
    print(f"\nRegistry saved to {REGISTRY_PATH.relative_to(ROOT)}")


def _meta_is_complete(source: Path) -> bool:
    """Return True if the .meta for *source* has no ``not_checked`` mutants."""
    meta = _meta_path_for(source)
    if not meta.exists():
        return False
    counts = _parse_meta(meta)
    return counts["not_checked"] == 0


def _restore_meta_backup(source: Path) -> bool:
    """Restore ``.meta.bak`` → ``.meta``.  Returns True if restored."""
    rel = source.relative_to(ROOT)
    backup = MUTANTS_DIR / f"{rel}.meta.bak"
    if backup.exists():
        meta = MUTANTS_DIR / f"{rel}.meta"
        backup.rename(meta)
        return True
    return False


def _remove_meta_backup(source: Path) -> None:
    """Delete the ``.meta.bak`` file if it exists."""
    rel = source.relative_to(ROOT)
    backup = MUTANTS_DIR / f"{rel}.meta.bak"
    if backup.exists():
        backup.unlink()


def run_batch(targets: list[str], clean: bool = True) -> None:
    """Run mutmut on each .py file individually, updating registry after each."""
    sources: list[Path] = []
    for target in targets:
        sources.extend(_source_files(target))

    if not sources:
        sys.exit("No .py files found for the given targets.")

    # Load registry once for freshness checks
    registry: dict[str, Any] = {}
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH) as f:
            registry = yaml.safe_load(f) or {}

    skipped = 0
    restored = 0
    print(f"Batch mode: {len(sources)} file(s) to process\n")

    for i, src in enumerate(sources, 1):
        rel = src.relative_to(ROOT)
        glob = _to_mutmut_glob(src)
        print(f"[{i}/{len(sources)}] {rel}")

        if _is_fresh(src, registry):
            print("  (fresh, skipped)\n")
            skipped += 1
            continue

        if clean:
            _clean_cache(src, backup_meta=True)

        cmd = [sys.executable, ENTRY_SCRIPT, "run", glob]
        timed_out = False
        try:
            subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=600)  # nosec B603
        except subprocess.TimeoutExpired:
            timed_out = True

        if timed_out or not _meta_is_complete(src):
            reason = "timed out" if timed_out else "incomplete results"
            if _restore_meta_backup(src):
                print(f"  ⚠ {reason}, restored previous results\n")
                restored += 1
            else:
                print(f"  ⚠ {reason}, no backup to restore\n")
            continue

        _remove_meta_backup(src)

        updated = _update_registry([src])
        if updated:
            e = list(updated.values())[0]
            if e["total"] == 0:
                print("  (no mutants)\n")
            else:
                print(
                    f"  {e['total']} mutants: "
                    f"{e['killed']} killed, {e['survived']} survived "
                    f"-> {e['score']:.1f}%\n"
                )
        else:
            print("  (no mutants)\n")

    if skipped:
        print(f"Skipped {skipped} fresh file(s).")
    if restored:
        print(f"Restored {restored} file(s) from backup (incomplete run).")

    print("\n" + "=" * 83)
    print("BATCH COMPLETE\n")
    report(None)


def report(targets: list[str] | None) -> None:
    """Show registry report, optionally filtered by targets."""
    if targets:
        sources: list[Path] = []
        for t in targets:
            sources.extend(_source_files(t))
        updated = _update_registry(sources)
        _print_table(updated)
    else:
        # Show full registry
        if not REGISTRY_PATH.exists():
            print("No registry found. Run mutmut on some modules first.")
            return
        with open(REGISTRY_PATH) as f:
            registry = yaml.safe_load(f) or {}
        _print_table(registry)


def main() -> None:
    """CLI entry point for per-module mutmut runs."""
    parser = argparse.ArgumentParser(
        description="Run mutmut on modules and update the mutation registry."
    )
    parser.add_argument(
        "targets",
        nargs="*",
        help="Source file(s) or directory(ies) to mutate (e.g. orchard/cli_app.py orchard/core/config/)",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Show registry report without running mutmut",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run each .py file individually (clean cache, update registry after each)",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Skip cache cleaning in batch mode (use existing trampolines)",
    )
    args = parser.parse_args()

    if args.report:
        report(args.targets if args.targets else None)
    elif not args.targets:
        parser.print_help()
        sys.exit(1)
    elif args.batch:
        run_batch(args.targets, clean=not args.no_clean)
    else:
        run_mutmut(args.targets)


if __name__ == "__main__":
    main()
