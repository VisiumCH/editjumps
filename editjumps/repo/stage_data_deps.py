"""Print the DVC-tracked data dependencies of the named stages, one per line.."""

import sys


def data_deps(stage_names: list[str], repo_root: str = ".") -> list[str]:
    """Resolve the cached data dependencies of the named stages, as named in ``dvc.yaml``."""
    from dvc.repo import Repo

    repo = Repo(repo_root)
    wanted = set(stage_names)
    by_name = {s.addressing: s for s in repo.index.stages}

    missing = wanted - set(by_name)
    if missing:
        raise SystemExit(f"unknown stage(s): {', '.join(sorted(missing))}")

    # A dep is pullable iff some stage produces it or a .dvc file tracks it — i.e. it is in the
    # cache. Anything else (source files) must not be handed to `dvc pull`.
    produced = {str(out) for stage in repo.index.stages for out in stage.outs}

    # ...but NOT if a requested stage produces it in this same run: those artefacts are not in the remote yet.
    produced_here = {str(out) for name in wanted for out in by_name[name].outs}

    paths: set[str] = set()
    for name in wanted:
        for dep in by_name[name].deps:
            if str(dep) in produced and str(dep) not in produced_here:
                paths.add(dep.def_path)
    return sorted(paths)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: stage_data_deps.py <stage> [<stage>...]")
    print("\n".join(data_deps(sys.argv[1:])))
