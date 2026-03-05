<- [Back to Home](../index.md) | [Back to Testing](TESTING.md)

<h1 align="center">Mutation Testing</h1>

Orchard ML uses [mutmut v3](https://mutmut.readthedocs.io/) for mutation testing.
Mutmut injects small code changes (mutants) and verifies that the test suite
catches each one. Survived mutants indicate gaps in test assertions.

---

<h2>Configuration</h2>

Mutation testing is configured in `pyproject.toml`:

```toml
[tool.mutmut]
paths_to_mutate = ["orchard/"]
tests_dir = ["tests/"]
```

Cosmetic code (log formatting, Excel styling) is excluded via line-level
`# pragma: no mutate` annotations. These are **only** applied to
`logger.info()` and `logger.debug()` calls — never to warnings, errors,
or real logic.

---

<h2>Running Mutation Tests</h2>

**Full repository** (slow — hours on first run):

```bash
# Generate mutants and run tests against each one
mutmut run

# View results summary
mutmut results

# Inspect a specific survived mutant
mutmut show <mutant_name>
```

**Single module** (recommended for iterative work):

mutmut v3 uses dotted-module glob patterns as positional arguments:

```bash
# Mutate only the search_spaces module
mutmut run "orchard.optimization.search_spaces*"

# Mutate only the loader module
mutmut run "orchard.data_handler.loader*"

# Mutate only the evaluation pipeline
mutmut run "orchard.evaluation.evaluation_pipeline*"
```

**Multiple modules** in one run:

```bash
mutmut run "orchard.optimization*" "orchard.trainer*"
```

**Single class or function:**

```bash
mutmut run "orchard.optimization.search_spaces.*SearchSpaceRegistry*"
mutmut run "*get_optimization_space*"
```

---

<h2>Mutation Registry</h2>

The mutation registry (`mutmut-registry.yaml`) tracks per-file mutation scores
and auto-updates when you test a module. Use `scripts/mutmut_run.py`:

```bash
# Run mutmut on a single file and update the registry
python scripts/mutmut_run.py orchard/cli_app.py

# Run mutmut on an entire sub-package
python scripts/mutmut_run.py orchard/core/config/

# Multiple targets at once
python scripts/mutmut_run.py orchard/cli_app.py orchard/exceptions.py

# Show the registry report (no mutmut run, just read existing results)
python scripts/mutmut_run.py --report

# Show report for specific modules
python scripts/mutmut_run.py --report orchard/core/config/

# Batch: run each .py file one by one (cleans cache, updates registry after each)
python scripts/mutmut_run.py --batch orchard/trainer/

# Batch the whole project
python scripts/mutmut_run.py --batch orchard/
```

**Output example:**

```
Module                                                  Total  Kill  Surv   N/C   Score
---------------------------------------------------------------------------------------
orchard/cli_app.py                                         45    42     3     0   93.3%
orchard/exceptions.py                                       8     8     0     0  100.0%
---------------------------------------------------------------------------------------
TOTAL                                                      53    50     3     0   94.3%
```

The registry YAML is tracked in git so you can see score evolution across commits.

---

<h2>Cleaning Cache</h2>

mutmut v3 caches trampoline files and metadata in the `mutants/` directory.
It skips re-generation when the trampoline is newer than the source file.
To force a fresh run, delete **both** the trampoline and its metadata:

```bash
# Clean cache for a specific module
rm mutants/orchard/optimization/search_spaces.py \
   mutants/orchard/optimization/search_spaces.py.meta

# Alternative: touch the source file to invalidate the cache
touch orchard/optimization/search_spaces.py

# Clean all cached results
rm -rf mutants/
```

---

<h2>Filtering Results</h2>

```bash
# Show all results (killed + survived)
mutmut results --all true

# Show only survived mutants (the ones to fix)
mutmut results

# Inspect a specific survived mutant
mutmut show <mutant_name>
```

---

<h2>Writing Mutation-Resilient Tests</h2>

Tests that only check key presence (`assert "key" in space`) will let many
mutants survive. To kill mutants effectively:

- **Assert exact values** passed to functions (bounds, lists, constants)
- **Assert exact return values**, not just types
- **Test boundary conditions** (e.g., resolution 223 vs 224)
- **Test both branches** of conditionals (enabled/disabled, present/absent)
- **Verify side effects** (function called vs not called)

---

<h2>Pragma Conventions</h2>

| Annotation | Scope | Usage |
|---|---|---|
| `# pragma: no mutate` | Single line | `logger.info()`, `logger.debug()`, plot formatting |
| `# pragma: no cover` | Single line | Unreachable defensive code |

**Never** apply `# pragma: no mutate` to:

- `logger.warning()` or `logger.error()` calls
- `warnings.warn()` calls
- Conditionals, computed values, or any real logic
- Entire files (`do_not_mutate` is forbidden)

---

<h2>conftest Helper</h2>

When tests use `patch.dict(os.environ, ..., clear=True)`, mutmut v3
trampolines break because `MUTANT_UNDER_TEST` is wiped. Use the
`mutmut_safe_env()` helper from `tests/conftest.py`:

```python
from tests.conftest import mutmut_safe_env

def test_something():
    with patch.dict(os.environ, mutmut_safe_env(MY_VAR="1"), clear=True):
        ...
```

---
