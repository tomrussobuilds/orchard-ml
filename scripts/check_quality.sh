#!/bin/bash
# Quality checks automation script
# Run all code quality checks in one go

set -e

echo "🔍 VisionForge Quality Checks"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "📝 Black (code formatting)..."
black --check --diff orchard/ tests/ main.py optimize.py
echo "✓ Black passed"
echo ""

echo "📦 isort (import sorting)..."
isort --check-only --diff orchard/ tests/ main.py optimize.py
echo "✓ isort passed"
echo ""

echo "✨ Flake8 (linting)..."
flake8 orchard/ tests/ main.py optimize.py --max-line-length=100 --extend-ignore=E203,W503
echo "✓ Flake8 passed"
echo ""

echo "🧪 Pytest (tests + coverage)..."
pytest --cov=orchard --cov-report=term-missing -v tests/
echo ""

echo "✅ All quality checks passed!"
