#!/usr/bin/env python3
"""Strip outputs from Jupyter notebooks before commit."""

import json
import sys


def strip_outputs(path: str) -> bool:
    """Remove outputs and execution counts. Return True if file was modified."""
    with open(path) as f:
        nb = json.load(f)

    modified = False
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            if cell["outputs"]:
                cell["outputs"] = []
                modified = True
            if cell["execution_count"] is not None:
                cell["execution_count"] = None
                modified = True

    if modified:
        with open(path, "w") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
            f.write("\n")

    return modified


if __name__ == "__main__":
    for filepath in sys.argv[1:]:
        if strip_outputs(filepath):
            print(f"Stripped outputs: {filepath}")
