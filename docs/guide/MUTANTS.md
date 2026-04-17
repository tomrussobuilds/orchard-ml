← [Back to Home](../index.md) | [Back to Testing](TESTING.md)

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

Log and cosmetic mutations are suppressed **automatically** by the patched
entry point `scripts/mutmut_entry.py` — no per-line `# pragma: no mutate`
annotations are needed for logging calls.  See [Patched Entry Point](#patched-entry-point)
below for details.

---

<h2>Running Mutation Tests</h2>

> [!WARNING]
> **Prerequisites**
>
> 1. **Always use `.venv/bin/python`** — never system python.
> 2. **All tests must pass before running mutmut.** A single test failure
>    causes ALL mutants to be marked `not_checked`, and batch mode sees
>    "incomplete results" and skips/restores backup.
>
> ```bash
> .venv/bin/python -m pytest tests/ -x -q
> ```

**Full repository** (slow — hours on first run):

```bash
# Generate mutants and run tests against each one
.venv/bin/python scripts/mutmut_entry.py run

# View results summary
.venv/bin/python scripts/mutmut_entry.py results

# Inspect a specific survived mutant
.venv/bin/python scripts/mutmut_entry.py show <mutant_name>
```

**Single module** (recommended for iterative work):

mutmut v3 uses dotted-module glob patterns as positional arguments:

```bash
# Mutate only the search_spaces module
.venv/bin/python scripts/mutmut_entry.py run "orchard.optimization.search_spaces*"

# Mutate only the loader module
.venv/bin/python scripts/mutmut_entry.py run "orchard.data_handler.loader*"

# Mutate only the evaluation pipeline
.venv/bin/python scripts/mutmut_entry.py run "orchard.evaluation.evaluation_pipeline*"
```

**Multiple modules** in one run:

```bash
.venv/bin/python scripts/mutmut_entry.py run "orchard.optimization*" "orchard.trainer*"
```

**Single class or function:**

```bash
.venv/bin/python scripts/mutmut_entry.py run "orchard.optimization.search_spaces.*SearchSpaceRegistry*"
.venv/bin/python scripts/mutmut_entry.py run "*get_optimization_space*"
```

> [!NOTE]
> Always use `scripts/mutmut_entry.py` instead of bare `mutmut` — the patched
> entry point suppresses cosmetic mutations on logging calls automatically.
> `scripts/mutmut_run.py` invokes it internally.

---

<h2>Mutation Registry</h2>

The mutation registry (`mutmut-registry.yaml`) tracks per-file mutation scores
and auto-updates when you test a module. Use `scripts/mutmut_run.py`:

```bash
# Run mutmut on a single file and update the registry
.venv/bin/python scripts/mutmut_run.py orchard/cli_app.py

# Run mutmut on an entire sub-package
.venv/bin/python scripts/mutmut_run.py orchard/core/config/

# Multiple targets at once
.venv/bin/python scripts/mutmut_run.py orchard/cli_app.py orchard/exceptions.py

# Show the registry report (no mutmut run, just read existing results)
.venv/bin/python scripts/mutmut_run.py --report

# Show report for specific modules
.venv/bin/python scripts/mutmut_run.py --report orchard/core/config/

# Batch: run each .py file one by one (cleans cache, updates registry after each)
.venv/bin/python scripts/mutmut_run.py --batch orchard/trainer/

# Batch the whole project
.venv/bin/python scripts/mutmut_run.py --batch orchard/
```

**Output example:**

```
Module                                                  Total  Kill  Surv   N/C   Score
---------------------------------------------------------------------------------------
orchard/architectures/factory.py                           80    80     0     0  100.0%
orchard/cli_app.py                                        507   477    30     0   94.1%
orchard/core/environment/hardware.py                      133   129     4     0   97.0%
---------------------------------------------------------------------------------------
TOTAL                                                     720   686    34     0   95.3%
```

The registry YAML is tracked in git so you can see score evolution across commits.

**Registry guards** (`scripts/check_mutmut_registry.py`):

```bash
# Fail if any module score dropped vs HEAD (pre-commit gate)
.venv/bin/python scripts/check_mutmut_registry.py --ratchet

# Fail if any modified module has a stale registry entry (release gate)
.venv/bin/python scripts/check_mutmut_registry.py --freshness

# Both
.venv/bin/python scripts/check_mutmut_registry.py --ratchet --freshness
```

---

<h2>Cleaning Cache</h2>

mutmut v3 caches trampoline files and metadata in the `mutants/` directory.
Always clean the **entire** `mutants/` directory before reruns — deleting
individual files is error-prone and can leave stale state:

```bash
# Clean all cached results (recommended)
rm -rf mutants/
```

> [!WARNING]
> **Uncommitted files and the registry**
>
> `--batch` mode uses `_is_fresh` which compares the registry `last_run`
> timestamp against `git log -1 --format=%aI`. **Uncommitted changes don't
> update `git log`**, so old registry entries look "newer" and the file gets
> **skipped silently**.
>
> Before running mutmut on uncommitted files, remove their registry entries
> **and** the cache:
>
> ```bash
> rm -rf mutants/
> .venv/bin/python -c "
> import yaml; from pathlib import Path
> reg_path = Path('mutmut-registry.yaml')
> reg = yaml.safe_load(reg_path.read_text()) or {}
> for k in ['orchard/path/to/changed_file.py']:
>     reg.pop(k, None)
> reg = dict(sorted(reg.items()))
> reg_path.write_text(yaml.dump(reg, default_flow_style=False, sort_keys=False))
> "
> ```

---

<h2>Gotchas</h2>

> [!CAUTION]
> **Never use `--batch` on `__init__.py` files**
>
> `_to_mutmut_glob` strips `.__init__` and appends `*`, so
> `orchard/__init__.py` becomes glob `orchard*` — which matches the
> **entire codebase**. Use `--report` instead for `__init__.py` and
> pure-declaration files (constants, re-exports) with no mutable logic:
>
> ```bash
> .venv/bin/python scripts/mutmut_run.py --report orchard/__init__.py orchard/tasks/__init__.py
> ```

> [!NOTE]
> **Batch timeout**
>
> Batch mode has a **600-second (10 min) timeout per file**. If exceeded,
> previous results are restored from the `.meta.bak` backup.

> [!NOTE]
> **CI does not run mutmut**
>
> Mutation testing is a **local quality gate only**. CI runs linting, type
> checking, and pytest — but not mutmut.

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

<h2>Patched Entry Point</h2>

`scripts/mutmut_entry.py` monkey-patches mutmut's `MutationVisitor` to
suppress cosmetic mutations without per-line annotations.  It is invoked
automatically by `scripts/mutmut_run.py`.

Two suppression levels:

| Level | Methods | Effect |
|---|---|---|
| **Full skip** | `debug`, `info`, `add_format` | Entire `Call` node excluded — call, arguments, and strings |
| **String-only skip** | `warning`, `error`, `warn`, `getLogger` | Only string literals inside the call are excluded; the call itself and non-string args remain mutable |

This eliminates the need for `# pragma: no mutate` on logging lines.

---

<h2>Pragma Conventions</h2>

| Annotation | Scope | Usage |
|---|---|---|
| `# pragma: no mutate` | Single line | Plot formatting constants, cosmetic-only literals |
| `# pragma: no cover` | Single line | Unreachable defensive code |

Logging calls (`info`, `debug`, `warning`, `error`, `warn`) are handled
automatically by the patched entry point — **do not** annotate them manually.

**Never** apply `# pragma: no mutate` to:

- Conditionals, computed values, or any real logic
- Entire files (`do_not_mutate` is forbidden)

---

<h2>Resolved Issue: `set_start_method` Crash</h2>

mutmut 3.5.0 calls `multiprocessing.set_start_method('fork')` at module level
in `mutmut/__main__.py`. When the module is re-executed (e.g. via
`python -m mutmut run`), the call fails with:

```
RuntimeError: context has already been set
```

**Status:** fixed upstream in [GH-466](https://github.com/boxed/mutmut/pull/466)
(merged into `main`). The fix guards the call with `get_start_method(allow_none=True)`
and is included in mutmut **> 3.5.0**. If you are still on 3.5.0, either install
from git:

```bash
uv pip install "mutmut @ git+https://github.com/boxed/mutmut.git@main"
```

or apply the local patch:

```bash
sed -i "s/set_start_method('fork')/set_start_method('fork', force=True)/" \
    .venv/lib/python3.*/site-packages/mutmut/__main__.py
```

---

<h2>Resolved Issue: Name Mangling in Trampoline Generation</h2>

When a class name starts with an underscore (e.g. `_CrossDomainValidator`),
mutmut generates trampoline function names like
`__CrossDomainValidator_validate_trampoline`. Inside the class body, Python's
[name mangling](https://docs.python.org/3/reference/expressions.html#atom-identifiers)
rewrites `__CrossDomainValidator_validate_trampoline` to
`_CrossDomainValidator__CrossDomainValidator_validate_trampoline`, causing a
`NameError` at import time.

**Status:** fixed upstream in [boxed/mutmut#499](https://github.com/boxed/mutmut/pull/499)
(merged 2026-04-16, reported in [#498](https://github.com/boxed/mutmut/issues/498)).
The fix uses a `_mutmut_` prefix instead of `_{class_name}_`, which is always
safe regardless of class name. No local patch needed once you install a release
that includes this fix.

If you are still on a build that predates the fix, apply the patch manually:

```bash
sed -i 's/prefix = f"_{class_name}_{method_name}"/prefix = f"_mutmut_{class_name}_{method_name}"/' \
    .venv/lib/python3.*/site-packages/mutmut/mutation/trampoline_templates.py \
    .venv/lib/python3.*/site-packages/mutmut/mutation/file_mutation.py
```

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
