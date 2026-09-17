# Installation

Setup and dependency configuration.

## Contents

- [Prerequisites](#prerequisites)
- [The Python environment](#the-python-environment)
- [Dependency groups](#dependency-groups)
- [External binaries that pip cannot install](#external-binaries-that-pip-cannot-install)
- [The SkyPilot CLI (`sky`)](#the-skypilot-cli-sky)
- [Isolated environments for third-party models](#isolated-environments-for-third-party-models)
- [Verifying the install](#verifying-the-install)
- [Getting the data](#getting-the-data)

## Prerequisites

| Requirement | Details |
|---|---|
| [`uv`](https://docs.astral.sh/uv/) | Manages Python environments and dependencies. Python does not need to be installed separately. |
| Python >=3.10 | Required minimum version specified in `pyproject.toml`. CI runs on Python 3.14. |
| `git` | Required by DVC. |
| HMMER (`hmmscan`) | Optional. Only used by standalone measurement scripts; not required for pipeline execution. |
| SkyPilot (`sky`) | Optional. Required for cloud execution targets (see [`docs/gcp_setup.md`](gcp_setup.md)). |

Local execution does not require a GPU. Pipeline preprocessing, small-scale local training, and evaluation run on standard CPU workstations.

## The Python environment

```bash
git clone <this repository> && cd editjumps
uv sync
```

This installs base dependencies and the `editjumps` CLI tool into `.venv/bin`.

```bash
uv run editjumps --help     # list pipeline subcommands
make help                  # list available targets
```

## Dependency groups

`pyproject.toml` defines four dependency groups. A default `uv sync` installs the `dev` group:

| Group | Dependencies | Used for |
|---|---|---|
| `dev` | ruff, ty, pytest, pre-commit, `dvc[gs]` | Linting, tests, and DVC pipeline management |
| `train` | torch, transformers, datasets, accelerate, fair-esm | Training, sequence editing, and generation scoring |
| `analysis` | fair-esm, umap-learn | Optional exploration and plotting |
| `notebook` | jupyterlab | Interactive notebooks |

To install all groups including PyTorch:

```bash
uv sync --all-groups
```

## External binaries that pip cannot install

Two external tools are required for specific parts of the pipeline:

### `hmmscan` (HMMER)

Optional. `editjumps/core/cdr.py` uses ANARCI to number CDRs via `hmmscan`. Only standalone analysis scripts invoke this module; pipeline stages, training, and tests do not require HMMER.

```bash
brew install hmmer          # macOS
apt-get install hmmer       # Debian/Ubuntu
```

### MMseqs2

Required for clustering stages (`split_corpus`, `build_homolog_pairs`). Install into `.venv/bin` after running `uv sync`:

```bash
bash editjumps/core/install_mmseqs.sh
```

This downloads a standalone static binary matching the host platform.

## The SkyPilot CLI (`sky`)

Cloud execution targets (under the `gpu-` and `jobs-` Makefile namespaces) use SkyPilot to provision and orchestrate cloud GPU instances. Install SkyPilot independently of the project virtualenv:

```bash
uv tool install 'skypilot[gcp]'
sky check gcp
```

SkyPilot is managed as a standalone tool to avoid constraining core dependency versions in `pyproject.toml`.

## Isolated environments for third-party models

Baseline comparisons against EvoDiff require an isolated environment due to incompatible dependencies (`evodiff` requires `numpy<2`):

```bash
make install-evodiff        # creates .evodiff_env and links runner to .venv/bin
```

## Verifying the install

```bash
uv run editjumps --help     # verify CLI is accessible
make check                  # run ruff and ty type checks
make test                   # run test suite
```

> [!NOTE]
> `make ci` executes `uv sync --locked --all-extras --dev` before testing, resetting the environment to development dependencies. If running training workflows subsequently, re-run `uv sync --all-groups`.

## Getting the data

Data artifacts are tracked via DVC. If cloud storage credentials are unavailable, rebuild datasets directly from public sources:

```bash
uv run dvc repro            # reproduces outputs from public OAS sources
```

Memory-intensive stages (`split_corpus`, `build_homolog_pairs`) can be executed on cloud GPU instances or run locally with explicit memory overrides. If you want to run locally thoses stages, ensure tool have a machine that has at least 64 GB of RAM.

To run it on the cloud:
```bash
make jobs-repro STAGES=split_corpus                   # remote execution
```
To run it locally:
```bash
EDITJUMPS_ALLOW_LOCAL_HEAVY=1 make repro STAGES=       # local execution override
```

## Where to go next

| Document | Description |
|---|---|
| [`weights.md`](weights.md) | Checkpoints and model weights |
| [`performance.md`](performance.md) | Resource profiles and runtime benchmarks |
| [`reproducing.md`](reproducing.md) | Reproduction workflows and comparability controls |
| [`gcp_setup.md`](gcp_setup.md) | Cloud infrastructure and instance configuration |
| [`known_issues.md`](known_issues.md) | Diagnostic symptoms and operational caveats |
