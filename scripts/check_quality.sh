#!/bin/bash
# Quality checks automation script
# Run all code quality checks in one go

set -e

echo "🔍 Orchard ML Quality Checks"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "📝 Black (code formatting)..."
black --check --diff orchard/ tests/
echo "✓ Black passed"
echo ""

echo "✨ Ruff (linting + import sorting)..."
ruff check orchard/ tests/
echo "✓ Ruff passed"
echo ""

echo "🔒 Bandit (security linting)..."
bandit -c pyproject.toml -r orchard/ -l -q
echo "✓ Bandit passed"
echo ""

echo "🔍 MyPy --strict (type checking)..."
rm -rf .mypy_cache
mypy --strict orchard/ tests/
echo "✓ MyPy --strict passed"
echo ""

echo "📊 Radon (complexity analysis)..."
echo "  Cyclomatic Complexity (max: B):"
radon cc orchard/ -n B --total-average
echo ""
echo "  Maintainability Index (min: B):"
radon mi orchard/ -n B
echo "✓ Radon passed"
echo ""

echo "🧪 Pytest (tests + coverage)..."
pytest --cov=orchard --cov-report=term-missing --cov-fail-under=100 -v tests/
echo ""

echo "✅ All quality checks passed!"
