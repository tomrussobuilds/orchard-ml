"""
Test Suite for the recipes shipped in the repository.

Every YAML under ``recipes/`` is parsed through ``Config.from_recipe`` so a
renamed or removed config field cannot silently invalidate the examples the
README and the guides tell users to run. All sub-configs declare
``extra="forbid"``, so a stale key is a hard failure rather than a warning.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from orchard.core.config.manifest import Config
from orchard.core.paths import PROJECT_ROOT

RECIPES_DIR = PROJECT_ROOT / "recipes"

# Ships as the counter-example for `orchard validate`: it carries deliberate
# errors (a misspelled optimiser_type, a renamed optuna key, an unknown
# sampler) and is asserted to fail rather than excluded from the sweep.
INVALID_RECIPES = frozenset({"test_invalid.yaml"})

_ALL_RECIPES = sorted(RECIPES_DIR.glob("*.yaml")) if RECIPES_DIR.is_dir() else []
VALID_RECIPES = [p for p in _ALL_RECIPES if p.name not in INVALID_RECIPES]
BROKEN_RECIPES = [p for p in _ALL_RECIPES if p.name in INVALID_RECIPES]


@pytest.mark.unit
class TestShippedRecipes:
    """Contract tests over the recipe corpus in the repository root."""

    def test_recipe_corpus_is_discovered(self) -> None:
        """
        The sweep below covers real files.

        Without this guard an empty or moved ``recipes/`` directory would
        collapse every parametrised case into zero tests and still report
        green.
        """
        if not RECIPES_DIR.is_dir():
            pytest.skip("recipes/ not present (package installed without it)")

        assert VALID_RECIPES, f"no recipes discovered under {RECIPES_DIR}"

    @pytest.mark.parametrize("recipe", VALID_RECIPES, ids=lambda p: p.name)
    def test_shipped_recipe_is_valid(self, recipe: Path) -> None:
        """
        Each shipped recipe loads into a Config without overrides.
        """
        Config.from_recipe(recipe)

    @pytest.mark.parametrize("recipe", BROKEN_RECIPES, ids=lambda p: p.name)
    def test_intentionally_invalid_recipe_is_rejected(self, recipe: Path) -> None:
        """
        The `orchard validate` counter-example still fails validation.
        """
        with pytest.raises(ValidationError):
            Config.from_recipe(recipe)
