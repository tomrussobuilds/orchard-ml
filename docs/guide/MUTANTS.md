← [Back to Home](../index.md) | [Back to Testing](TESTING.md)

<h1 align="center">Mutation Testing</h1>

Orchard ML uses [mutmut v3](https://mutmut.readthedocs.io/) for mutation testing.
Mutmut injects small code changes (mutants) and verifies that the test suite
catches each one. Survived mutants indicate gaps in test assertions.

---

<h2>Installing mutmut</h2>

> [!IMPORTANT]
> **mutmut is pinned to an exact commit of git `main`, not to a PyPI release.**
>
> ```toml
> "mutmut @ git+https://github.com/boxed/mutmut.git@cd2f73da310c3fc90cffcb3e6c768cdeac14e18c"
> ```
>
> No published release works. What the pinned commit carries:
>
> | Change | Available in |
> |---|---|
> | `mutmut.mutation.*` layout, imported by `scripts/mutmut_entry.py` | 3.6.0 onwards |
> | `set_start_method` guard ([GH-466](https://github.com/boxed/mutmut/pull/466)) | after 3.5.0 |
> | env scrubbing, `MUTANT_UNDER_TEST` ([#511](https://github.com/boxed/mutmut/issues/511)) | **after 3.7.0, `main` only** |
> | `fix: mutate methods of decorated classes` ([#539](https://github.com/boxed/mutmut/pull/539)) | **after 3.7.0, `main` only** |
>
> So 3.5.0 is unusable (wrong layout) and 3.7.0 is the newest release yet still
> ships the #511 bug: its trampoline does `os.environ.get("MUTANT_UNDER_TEST", "")`
> on every call, verified directly on the wheel.
>
> The pin is an exact SHA, not `@main`, and that is not pedantry: mutant
> *generation* changes between commits, so the registry ratchet only compares
> like with like while the revision is frozen. Moving 3.5.0 → `cd2f73d` took
> `search_spaces.py` from 327 mutants to 358. Treat a mutmut bump like a `mypy`
> bump: deliberate, and followed by a re-baseline of the affected modules.
>
> After rebuilding the venv, `uv sync` restores the pinned commit. To install it
> by hand:
>
> ```bash
> uv pip install --no-deps \
>   "mutmut @ git+https://github.com/boxed/mutmut.git@cd2f73da310c3fc90cffcb3e6c768cdeac14e18c"
> ```
>
> `--no-deps` is deliberate: full resolution hits PyPI and has timed out on
> `setproctitle`. Every mutmut dependency is already in the venv.

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

# Batch the whole project (directory expansion skips __init__.py, see Gotchas)
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
> **Never name an `__init__.py` explicitly in `--batch`**
>
> `_to_mutmut_glob` strips `.__init__` and appends `*`, so
> `orchard/__init__.py` becomes glob `orchard*` — which matches the
> **entire codebase**, in a single batch step with a 600 s timeout.
>
> Directory targets are safe: `_source_files` skips `__init__.py` when
> expanding a directory, so `--batch orchard/` cannot trip this. The hazard is
> only in passing one by name. Use `--report` for `__init__.py` and other
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

> [!CAUTION]
> **Decorated functions produce zero mutants**
>
> mutmut does not wrap a decorated function or method in a trampoline, so its
> body is never mutated. Verified on the pinned commit:
>
> ```
> training_config.py   1 trampoline  (the free function is_amp_safe_batch_size)
>                      0             (both @model_validator methods)
> cli_app.py           0             (run / init / validate / main, @app.command)
> ```
>
> The consequence is that **Pydantic validators and Typer commands are not
> mutation tested at all**. Ten non-trivial files sit at `total: 0`, essentially
> the whole config layer. Read a 100 % score on a config module as "nothing was
> measured", not as "fully covered".
>
> This is distinct from [#539](https://github.com/boxed/mutmut/pull/539) and
> [#558](https://github.com/boxed/mutmut/issues/558), which cover decorated
> *classes*. Not yet reported upstream.

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
(merged into `main`) and present in the installed build. See
[Installing mutmut](#installing-mutmut) — the `sed` patch that used to be
documented here is obsolete, and would not have helped anyway: on PyPI 3.5.0
the blocker is the package layout, not that one line.

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
The pinned commit generates prefixes of the form
`x{CLASS_NAME_SEPARATOR}{class_name}{CLASS_NAME_SEPARATOR}`, which never starts
with a double underscore and so is immune to name mangling. No local patch is
needed, and the `sed` recipe once documented here no longer matches anything.

---

<h2>Resolved Issue: env scrubbing wipes `MUTANT_UNDER_TEST`</h2>

When tests use `patch.dict(os.environ, ..., clear=True)`, mutmut v3
trampolines break because `MUTANT_UNDER_TEST` is wiped from the environment.
Mutants reachable only through such tests are falsely reported as `survived`,
silently lowering the mutation score (measured impact on
`orchard/core/environment/hardware.py`: 19 false survivors, 85.7 % vs. the
true 97.0 %).

**Status:** fixed upstream. Reported as
[boxed/mutmut#511](https://github.com/boxed/mutmut/issues/511), closed as
completed on 2026-08-17 via [#552](https://github.com/boxed/mutmut/pull/552).
The trampoline now keeps the mutant name in module state
(`mutation/trampoline.py`) and only *prefers* the environment variable when
present, so scrubbing `os.environ` no longer deactivates the mutant. The fix
landed **after** the 3.7.0 tag, so it exists only on `main`: verified present in
the pinned commit `cd2f73d`, and verified *absent* from PyPI 3.7.0.

**Workaround removed (2026-09-03).** The `mutmut_safe_env()` helper that
re-injected `MUTANT_UNDER_TEST` into the patched env is gone, along with its 17
call sites in `tests/test_environment/test_hardware.py` and
`tests/test_paths/test_constants.py`. Verified on the original repro: without
the helper, `orchard/core/environment/hardware.py` scores **97.7 %**
(133 mutants, 3 survivors) instead of the 85.7 % the bug used to cause.

Scrubbing the environment in a test is safe again:

```python
def test_something():
    with patch.dict(os.environ, {"MY_VAR": "1"}, clear=True):
        ...
```

---

<h2>Pending Issue: mutants leak into `Protocol` class namespaces</h2>

mutmut emits its mutant variants **inside** the class body. For a
`runtime_checkable` `Protocol`, those names do not start with an underscore, so
`typing._get_protocol_attrs` counts them as protocol members:

```python
# in the mutated tree
_get_protocol_attrs(TaskEvalPipeline)
['run_evaluation',
 'xǁTaskEvalPipelineǁrun_evaluation__mutmut_1',
 'xǁTaskEvalPipelineǁrun_evaluation__mutmut_2',
 'xǁTaskEvalPipelineǁrun_evaluation__mutmut_orig']
```

Every `isinstance(obj, SomeProtocol)` check then fails, which aborts the stats
run and marks **all** mutants `not_checked` — for every module, not just the one
being measured.

**Status:** not yet reported upstream (observed on the pinned commit `cd2f73d`).

**Workaround:** `scripts/mutmut_entry.py` skips class definitions inheriting
from `Protocol` (`_is_protocol_class`). Protocol methods are declaration-only
stubs, so no meaningful mutant is lost.

---
