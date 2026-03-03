#!/bin/bash
# Extended quality checks with type checking and deep analysis
# More thorough but slower than check_quality.sh

set -e

echo "🔍 Orchard ML Extended Quality Checks"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
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
mypy --strict orchard/
echo "✓ MyPy --strict passed"
echo ""

echo "📊 Radon (complexity analysis)..."
echo "  Cyclomatic Complexity (max: B):"
radon cc orchard/ -n B --total-average
echo ""
echo "  Maintainability Index (min: B):"
radon mi orchard/ -n B
echo ""
echo "  Raw Metrics:"
radon raw orchard/ -s
echo "✓ Radon passed"
echo ""

echo "🧪 Pytest (tests + coverage)..."
pytest --cov=orchard --cov-report=term-missing --cov-report=html --cov-fail-under=100 -v tests/
echo ""

echo "✅ All extended quality checks passed!"
echo ""
echo "📊 Coverage report: htmlcov/index.html"
