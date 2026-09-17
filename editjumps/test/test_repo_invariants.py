"""Repo-wide invariants: provenance labels, the Makefile, and every internal link."""

from pathlib import Path


def test_provenance_classifies_every_dvc_stage_exactly() -> None:
    """Every pipeline stage carries a claim from CLAIMS, and nothing extra is mapped."""
    import yaml

    from editjumps.core.provenance import CLAIMS, PAPER_CLAIMS, STAGE_PROVENANCE

    stages = set(yaml.safe_load(Path("dvc.yaml").read_text())["stages"])
    mapped = set(STAGE_PROVENANCE)

    assert not stages - mapped, f"unclassified dvc stages: {sorted(stages - mapped)}"
    assert not mapped - stages, f"provenance entries for non-existent stages: {sorted(mapped - stages)}"
    assert all(p.claim in CLAIMS for p in STAGE_PROVENANCE.values())
    # A deviation is only meaningful against a paper decision, so it is allowed on any claim that reproduces a.
    assert PAPER_CLAIMS <= set(CLAIMS), f"PAPER_CLAIMS names a claim that does not exist: {PAPER_CLAIMS - set(CLAIMS)}"
    assert not [n for n, p in STAGE_PROVENANCE.items() if p.deviation and p.claim not in PAPER_CLAIMS]
    assert all(p.why.strip() for p in STAGE_PROVENANCE.values()), "every entry needs a justification"


def test_makefile_recipes_go_through_the_editjumps_cli() -> None:
    """No Makefile recipe may call a `python -m editjumps.<...>` path directly. editjumps/main.py is the."""
    import re

    recipes = [
        line for line in Path("Makefile").read_text().splitlines()
        if line.startswith("\t")
    ]
    offenders = [line.strip() for line in recipes if re.search(r"python -m editjumps\.", line)]
    assert not offenders, (
        "Makefile recipes must call `editjumps <subcommand>`, not a module path:\n  "
        + "\n  ".join(offenders)
    )


def test_every_makefile_editjumps_subcommand_exists() -> None:
    """Every `editjumps <cmd>` a Makefile target invokes is actually registered."""
    import re

    from editjumps.main import app

    # typer's registry, not click internals: `get_command(app)` is typed as Command, whose
    # `.commands` exists only on the group it returns.
    registered = {command.name for command in app.registered_commands}
    invoked = set(re.findall(r"editjumps ([a-z][a-z0-9-]+)", Path("Makefile").read_text()))
    missing = invoked - registered
    assert not missing, f"Makefile calls unregistered editjumps subcommands: {sorted(missing)}"


def test_every_make_target_referenced_in_scripts_and_docs_exists() -> None:
    """A renamed target leaves dead instructions behind, and only a reader discovers them."""
    import re

    targets = set(re.findall(r"^([a-z][a-z0-9-]*):", Path("Makefile").read_text(), re.M))
    assert "test" in targets, "sanity: the Makefile parsed"

    referenced: dict[str, set[str]] = {}
    shell = [*Path("editjumps").rglob("*.sh")]
    for path in [*shell, *Path(".").glob("*.md"), *Path("docs").rglob("*.md")]:
        # `.#README.md` is Emacs's lock symlink for an open buffer, and it dangles by design: it points at.
        if not path.is_file():
            continue
        text = path.read_text()
        pattern = r"`make ([a-z][a-z0-9-]+)" if path.suffix == ".md" else r"make ([a-z][a-z0-9-]+)"
        for hit in re.findall(pattern, text):
            referenced.setdefault(hit, set()).add(str(path))
        if path.suffix == ".md":            # fenced and indented blocks carry bare commands too
            for block in re.findall(r"```.*?```", text, re.S):
                for hit in re.findall(r"make ([a-z][a-z0-9-]+)", block):
                    referenced.setdefault(hit, set()).add(str(path))

    # Kept only for the shell scripts, where a bare `make X` is still matched.
    prose = {"real", "sense", "sure", "it", "them", "this", "the", "a", "an", "up", "no"}
    stale = {t: sorted(where) for t, where in referenced.items()
             if t not in targets and t not in prose}
    assert not stale, f"dead `make` instructions: {stale}"


def test_every_internal_markdown_link_resolves() -> None:
    """No relative link in a tracked Markdown page may point at a file that is not there."""
    import re

    root = Path(__file__).parents[2]
    # `rglob`, so docs/model_cards/ is covered too: those pages link sideways to each other and upward to.
    pages = sorted(root.glob("*.md")) + sorted((root / "docs").rglob("*.md"))
    dead = []
    for page in pages:
        # Emacs's `.#name.md` lock symlink dangles by design (it points at `user@host.pid:boot`), so
        # a glob over *.md finds a path `read_text` cannot open while a file is open in the editor.
        if not page.is_file():
            continue
        for label, target in re.findall(r"\[([^\]]+)\]\(([^)]+)\)", page.read_text()):
            if target.startswith(("http", "#", "mailto:")):
                continue
            if not (page.parent / target.split("#")[0]).resolve().exists():
                dead.append(f"{page.relative_to(root)}: [{label}]({target})")
    assert not dead, "dead internal links:\n" + "\n".join(dead)


def test_readme_hero_svgs_are_generated_and_differ_only_in_palette() -> None:
    """The two hero SVGs are a `make hero` product and must differ ONLY by colour."""
    import re
    import subprocess
    import sys

    root = Path(__file__).parents[2]
    images = root / ".github" / "images"
    light, dark = (images / "hero-light.svg").read_text(), (images / "hero-dark.svg").read_text()

    # Regenerating must be a no-op. If it is not, someone edited the SVGs instead of the generator.
    before = {p: p.read_bytes() for p in sorted(images.glob("hero-*.svg"))}
    subprocess.run([sys.executable, str(root / "editjumps" / "repo" / "make_hero.py")], check=True,
                   capture_output=True)
    assert {p: p.read_bytes() for p in sorted(images.glob("hero-*.svg"))} == before, (
        "hero SVGs are stale - run `make hero` and commit the result"
    )

    # Strip every colour and the two files must be byte-identical: same geometry, same text.
    colourless = [re.sub(r"#[0-9a-fA-F]{6}", "#", text) for text in (light, dark)]
    assert colourless[0] == colourless[1], "hero variants differ by more than their palette"

    # The README must reference both, or the dark one is dead weight.
    readme = (root / "README.md").read_text()
    assert ".github/images/hero-dark.svg" in readme
    assert ".github/images/hero-light.svg" in readme

    # The Feller portrait is the same deal: a themed pair, referenced from the README, differing only in.
    feller = [(images / f"feller-{theme}.svg").read_text() for theme in ("light", "dark")]
    assert re.sub(r"#[0-9a-fA-F]{6}", "#", feller[0]) == re.sub(r"#[0-9a-fA-F]{6}", "#", feller[1])
    assert ".github/images/feller-dark.svg" in readme
    assert ".github/images/feller-light.svg" in readme
    # The source photograph must never be committed: MacTutor states it does not hold its copyright.
    assert not (images / ".feller-source.jpg").exists() or ".feller-source.jpg" in (
        root / ".gitignore").read_text()
    # Feller is quoted, so he must be cited.
    assert "Berkeley Symposium" in readme and "Feller, W. (1949)" in readme
    # Static badges only: this repo is private, so anything shields.io has to READ renders broken.
    assert "shields.io/badge/" in readme
    assert "actions/workflows/ci.yml/badge.svg" not in readme.split("<!--")[0]


def test_every_measuring_stage_is_tracked_or_exempt() -> None:
    """A stage that measures something must open an MLflow run, or say in writing why it does not."""
    import re

    from editjumps.core.provenance import MLFLOW_EXEMPT

    root = Path(__file__).parents[2]
    # Both stage trees: `editjumps/pipeline/` holds the reproduction's stages and
    # entries below -- scanning only one tree would make every one of those entries read as stale.
    stage_trees = (root / "editjumps" / "pipeline",)
    measures = ("metrics_path", "metrics/", "figures/")
    untracked = []
    for path in sorted(p for tree in stage_trees for p in tree.rglob("*.py")):
        if path.name == "__init__.py":
            continue
        source = path.read_text()
        if not re.search(r"^def main\(", source, re.M):
            continue
        if not any(token in source for token in measures):
            continue
        if "start_mlflow_run" in source:
            continue
        if path.stem not in MLFLOW_EXEMPT:
            untracked.append(str(path.relative_to(root)))
    assert not untracked, (
        "these stages produce measurements, open no MLflow run, and are not in MLFLOW_EXEMPT:\n"
        + "\n".join(f"    {name}" for name in untracked)
        + "\nEither call start_mlflow_run, or add an entry saying why not."
    )

    # An exemption for a stage that now DOES track is stale and should be deleted, or the map
    # becomes a list of things that used to be true.
    stems = {p.stem for tree in stage_trees for p in tree.rglob("*.py")}
    assert not (set(MLFLOW_EXEMPT) - stems), (
        f"MLFLOW_EXEMPT names modules that do not exist: {set(MLFLOW_EXEMPT) - stems}"
    )
    for stem, reason in MLFLOW_EXEMPT.items():
        matches = [p for tree in stage_trees for p in tree.rglob(f"{stem}.py")]
        if matches and "start_mlflow_run" in matches[0].read_text():
            raise AssertionError(f"{stem} is exempt but now opens a run; drop its entry")
        assert len(reason) > 40, f"{stem}: an exemption needs a real reason, not a placeholder"


def test_every_markdown_table_is_well_formed() -> None:
    """No tracked Markdown table may have a ragged row, a missing delimiter, or a split cell."""
    import re

    root = Path(__file__).parents[2]
    delimiter = re.compile(r"^:?-{1,}:?$")

    def cells(line: str) -> list[str]:
        body = line.strip().replace(r"\|", "\x00")
        return [c.strip() for c in body.strip("|").split("|")]

    pages = (sorted(root.glob("*.md")) + sorted((root / "docs").rglob("*.md"))
             + sorted((root / "metrics").rglob("*.md")))
    bad = []
    for page in pages:
        if not page.is_file():  # Emacs `.#name.md` lock symlinks dangle by design
            continue
        lines = page.read_text().splitlines()
        rel = page.relative_to(root)
        fenced, i = False, 0
        while i < len(lines):
            if lines[i].strip().startswith("```"):
                fenced = not fenced
                i += 1
                continue
            if fenced or not lines[i].strip().startswith("|"):
                i += 1
                continue
            start = i
            while i < len(lines) and lines[i].strip().startswith("|"):
                i += 1
            block = lines[start:i]
            if len(block) < 2:
                bad.append(f"{rel}:{start + 1}: one-line table block")
                continue
            if not all(delimiter.match(c) for c in cells(block[1]) if c):
                bad.append(f"{rel}:{start + 2}: second row is not a delimiter row: {block[1][:70]!r}")
                continue
            width = len(cells(block[0]))
            for offset, row in enumerate(block):
                if len(cells(row)) != width:
                    bad.append(f"{rel}:{start + offset + 1}: {len(cells(row))} cells, "
                               f"header has {width}: {row[:70]!r}")
            # Prose between two runs of rows is the signature of a newline inside a cell.
            if i < len(lines) and lines[i].strip():
                j = i
                while j < len(lines) and lines[j].strip() and not lines[j].strip().startswith("|"):
                    j += 1
                if j < len(lines) and lines[j].strip().startswith("|"):
                    bad.append(f"{rel}:{i + 1}: prose interrupts a table (newline inside a "
                               f"cell?): {lines[i][:70]!r}")
    assert not bad, "malformed Markdown tables:\n" + "\n".join(bad)


def test_the_fast_structural_hook_covers_every_repo_invariant_file() -> None:
    """`.pre-commit-config.yaml`'s `structure` hook must name every `test_repo_*.py` file."""
    import yaml

    root = Path(__file__).parents[2]
    config = yaml.safe_load((root / ".pre-commit-config.yaml").read_text())
    hooks = [h for repo in config["repos"] for h in repo["hooks"] if h["id"] == "structure"]
    assert len(hooks) == 1, "expected exactly one `structure` hook"
    entry = hooks[0]["entry"]

    missing = sorted(
        path.relative_to(root).as_posix()
        for path in (root / "editjumps" / "test").glob("test_repo_*.py")
        if path.relative_to(root).as_posix() not in entry
    )
    assert not missing, (
        "these structural test files do not run at commit time; add them to the `structure` "
        "hook's entry in .pre-commit-config.yaml:\n" + "\n".join(missing)
    )


def test_every_markdown_heading_anchor_resolves() -> None:
    """A `#anchor` in a tracked link must name a heading that exists in the target page.."""
    import re

    def slug(heading: str) -> str:
        text = re.sub(r"[`*_~]", "", heading.strip())
        text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
        text = re.sub(r"[^\w\- ]", "", text.lower())
        return text.replace(" ", "-")

    def anchors(path: Path) -> set[str]:
        seen: dict[str, int] = {}
        found: set[str] = set()
        fenced = False
        for line in path.read_text().splitlines():
            if line.strip().startswith("```"):
                fenced = not fenced
                continue
            match = re.match(r"^(#{1,6})\s+(.*)$", line)
            if fenced or not match:
                continue
            base = slug(match.group(2))
            repeat = seen.get(base, 0)
            seen[base] = repeat + 1
            found.add(base if repeat == 0 else f"{base}-{repeat}")
        return found

    root = Path(__file__).parents[2]
    pages = (sorted(root.glob("*.md")) + sorted((root / "docs").rglob("*.md"))
             + sorted((root / "metrics").rglob("*.md")))
    cache: dict[Path, set[str]] = {}
    dead = []
    for page in pages:
        if not page.is_file():
            continue
        for label, target in re.findall(r"\[([^\]]+)\]\(([^)]+)\)", page.read_text()):
            if target.startswith(("http", "mailto:")) or "#" not in target:
                continue
            file_part, _, anchor = target.partition("#")
            if not anchor:
                continue
            dest = page if not file_part else (page.parent / file_part)
            if not dest.is_file():
                continue  # the file itself is the other test's business
            if dest not in cache:
                cache[dest] = anchors(dest)
            if anchor not in cache[dest]:
                dead.append(f"{page.relative_to(root)}: [{label}]({target})")
    assert not dead, "links to headings that do not exist:\n" + "\n".join(dead)


def test_no_inline_code_span_is_wrapped_through_an_identifier() -> None:
    """A wrapped `` `code span` `` must not break in the middle of an identifier."""
    import re

    root = Path(__file__).parents[2]
    pages = (sorted(root.glob("*.md")) + sorted((root / "docs").rglob("*.md"))
             + sorted((root / "metrics").rglob("*.md")))
    bad = []
    for page in pages:
        if not page.is_file():
            continue
        text = re.sub(r"^```.*?^```", lambda m: "\n" * m.group(0).count("\n"),
                      page.read_text(), flags=re.S | re.M)
        parts = text.split("`")
        if len(parts) % 2 == 0:
            bad.append(f"{page.relative_to(root)}: unbalanced ` in the page")
            continue
        offset = 0
        for index, part in enumerate(parts):
            if index % 2 == 1:
                for join in re.finditer(r"(?<! )([_.\-/=])\n\s*\w", part):
                    line = text.count("\n", 0, offset + join.start()) + 1
                    bad.append(f"{page.relative_to(root)}:{line}: `{part.strip()[:70]}`")
            offset += len(part) + 1
    assert not bad, "code spans wrapped through an identifier:\n" + "\n".join(bad)


def test_the_committed_claims_table_is_what_provenance_generates() -> None:
    """`docs/claims.md`'s generated block must equal what `readme_block()` renders. `provenance.py` said."""
    from editjumps.core.provenance import README_END, README_START, readme_block

    root = Path(__file__).parents[2]
    text = (root / "docs" / "claims.md").read_text()
    start, end = text.find(README_START), text.find(README_END)
    assert start != -1 and end != -1, "docs/claims.md has lost its provenance markers"
    assert text[start:end + len(README_END)] == readme_block(), (
        "docs/claims.md's generated block is stale; run `make provenance`."
    )
