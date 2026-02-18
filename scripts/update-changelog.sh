#!/usr/bin/env bash
# Regenerates CHANGELOG.md at every commit (pre-commit hook).
# git-cliff generates Unreleased + tagged releases after v0.1.0,
# then the curated v0.1.0 section is appended from changelog-base.md.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

git-cliff v0.1.0.. -o CHANGELOG.md
cat "$SCRIPT_DIR/changelog-base.md" >> CHANGELOG.md
git add CHANGELOG.md
