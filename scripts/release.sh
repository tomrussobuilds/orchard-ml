#!/usr/bin/env bash
# Usage: ./scripts/release.sh 0.2.0
#
# Bumps the version in pyproject.toml, commits, tags, and pushes.
# This triggers the release + publish workflows automatically.
set -euo pipefail

VERSION="${1:?Usage: $0 <version> (e.g. 0.2.0)}"
TAG="v${VERSION}"

# Validate semver format
if ! [[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "Error: version must be semver (e.g. 0.2.0)" >&2
    exit 1
fi

# Check for clean working tree
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "Error: working tree is not clean. Commit or stash changes first." >&2
    exit 1
fi

# Check tag doesn't already exist
if git rev-parse "$TAG" >/dev/null 2>&1; then
    echo "Error: tag $TAG already exists" >&2
    exit 1
fi

# Bump version in pyproject.toml
sed -i "s/^version = \".*\"/version = \"${VERSION}\"/" pyproject.toml
echo "Updated pyproject.toml to version ${VERSION}"

# Generate changelog with the new tag
git-cliff --tag "$TAG" -o CHANGELOG.md
echo "Generated CHANGELOG.md for ${TAG}"

# Commit and tag
git add pyproject.toml CHANGELOG.md
git commit -m "release: v${VERSION}"
git tag "$TAG"

# Push commit + tag
git push origin main
git push origin "$TAG"

echo ""
echo "Released ${TAG} — workflows triggered:"
echo "  - Release: creates GitHub Release with changelog"
echo "  - Publish: builds and publishes to PyPI"
