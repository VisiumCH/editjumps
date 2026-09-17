"""Which pipeline stages are tracked, and by what."""

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
#: A stage is treated as producing measurements if it writes a metrics path or a figure.
MEASURES = ("metrics_path", "metrics/", "figures/")


#: Where stage modules live.
STAGE_TREES = (ROOT / "editjumps" / "pipeline",)

#: Modules with a ``main`` that are NOT stages, so "not in the CLI" is correct for them rather than a gap.
NOT_STAGES = {
    "evodiff_msa_runner": "runs under .evodiff_env's interpreter, reached only via its PATH shim",
}


def stages() -> list[Path]:
    """Every pipeline module that defines a ``main``."""
    out = []
    for path in sorted(p for tree in STAGE_TREES for p in tree.rglob("*.py")):
        if path.name == "__init__.py" or path.stem in NOT_STAGES:
            continue
        if re.search(r"^def main\(", path.read_text(), re.M):
            out.append(path.relative_to(ROOT))
    return out


def cli_commands() -> dict[str, str]:
    """Map each ``editjumps <command>`` to the dotted module whose ``main`` it runs.

    Returns:
        CLI command name -> dotted module path, read out of `main.py`'s registrations.
    """
    tree = ast.parse((ROOT / "editjumps" / "main.py").read_text())
    alias_to_module: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("editjumps"):
            for alias in node.names:
                if alias.name == "main":
                    alias_to_module[alias.asname or "main"] = node.module
    commands: dict[str, str] = {}
    for node in ast.walk(tree):
        # `app.command("name", ...)(handler)`: a call whose callee is itself a call.
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Call)
                and isinstance(node.func.func, ast.Attribute) and node.func.func.attr == "command"
                and node.func.args and isinstance(node.func.args[0], ast.Constant)
                and len(node.args) == 1 and isinstance(node.args[0], ast.Name)):
            module = alias_to_module.get(node.args[0].id)
            if module:
                commands[str(node.func.args[0].value)] = module
    return commands


def dvc_modules() -> set[str]:
    """Return every module a ``dvc.yaml`` stage actually invokes, as a dotted path.

    Read out of the stage COMMANDS rather than the raw file. A substring test against the whole
    YAML matches stage names, deps, outs and comments too, so `evotune_baseline` counted as tracked
    on the strength of the unrelated `evotune_baseline_forced` stage name alone.

    Returns:
        Dotted module paths, resolved from both `python -m` and console-script invocations.
    """
    import yaml

    spec = yaml.safe_load((ROOT / "dvc.yaml").read_text()) or {}
    commands: list[str] = []
    for stage in (spec.get("stages") or {}).values():
        cmd = stage.get("cmd")
        commands.extend(cmd if isinstance(cmd, list) else [cmd or ""])
    joined = "\n".join(commands)

    modules = set(re.findall(r"python -m ([\w.]+)", joined))
    # `uv run editjumps <command>` names a CLI command rather than a module, so look the command up
    # in main.py to get the module it runs.
    registry = cli_commands()
    modules.update(registry[name] for name in re.findall(r"\beditjumps\s+([a-z0-9][a-z0-9-]*)", joined)
                   if name in registry)
    return modules


def main() -> None:
    """Print the coverage table and the gaps that matter."""
    tracked_modules = dvc_modules()
    cli = (ROOT / "editjumps" / "main.py").read_text()
    # Parsed with ast, not a regex: main.py has both single-line `from X import main as Y` and multi-line.
    registered = set()
    for node in ast.walk(ast.parse(cli)):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("editjumps"):
            if any(alias.name == "main" for alias in node.names):
                registered.add(node.module.replace(".", "/") + ".py")

    rows, missing_mlflow, missing_cli = [], [], []
    for path in stages():
        source = (ROOT / path).read_text()
        measures = any(token in source for token in MEASURES)
        tracked = "start_mlflow_run" in source
        in_dvc = path.with_suffix("").as_posix().replace("/", ".") in tracked_modules
        in_cli = str(path) in registered
        rows.append((str(path), measures, tracked, in_dvc, in_cli))
        if measures and not tracked:
            missing_mlflow.append(str(path))
        if not in_cli:
            missing_cli.append(str(path))

    print(f"{'stage':<64}{'measures':>9}{'mlflow':>8}{'dvc':>5}{'cli':>5}")
    print("-" * 91)
    for name, measures, tracked, in_dvc, in_cli in rows:
        print(f"{name:<64}{'yes' if measures else '-':>9}{'yes' if tracked else '-':>8}"
              f"{'yes' if in_dvc else '-':>5}{'yes' if in_cli else '-':>5}")

    print(f"\nPRODUCES MEASUREMENTS BUT OPENS NO MLFLOW RUN ({len(missing_mlflow)}):")
    for name in missing_mlflow:
        print(f"    {name}")
    print(f"\nHAS A `main` BUT IS NOT IN THE CLI ({len(missing_cli)}):")
    for name in missing_cli:
        print(f"    {name}")
    print(f"\nNOT STAGES, so absent from the CLI on purpose ({len(NOT_STAGES)}):")
    for stem, why in sorted(NOT_STAGES.items()):
        print(f"    {stem:<22} {why}")
    print("\nDVC coverage is enforced separately and in both directions by "
          "editjumps/core/provenance.py's test.")


if __name__ == "__main__":
    main()
