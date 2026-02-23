"""Generate the code reference pages and navigation."""

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()
root = Path("orchard")
reference_dir = "reference"

for path in sorted(root.rglob("*.py")):
    module_path = path.with_suffix("")
    doc_path = path.relative_to(root).with_suffix(".md")
    full_doc_path = Path(reference_dir, doc_path)

    parts = tuple(module_path.parts)

    # Skip private modules (except __init__.py which represents the package)
    if any(part.startswith("_") for part in parts[1:]):
        if parts[-1] != "__init__":
            continue

    # For __init__.py, use the package name
    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")

    if not parts:
        continue

    nav[parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        ident = ".".join(parts)
        fd.write(f"::: {ident}\n")

    mkdocs_gen_files.set_edit_path(full_doc_path, path)

with mkdocs_gen_files.open(f"{reference_dir}/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
