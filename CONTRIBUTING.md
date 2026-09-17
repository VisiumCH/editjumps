# Contributing

Guidelines for development, testing, and pull requests. [`.pre-commit-config.yaml`](.pre-commit-config.yaml), [`.github/workflows/ci.yml`](.github/workflows/ci.yml), and [`Makefile`](Makefile) represent authoritative configurations.

## 1. Environment Setup

[`uv`](https://docs.astral.sh/uv/) manages Python dependencies and virtual environments:

```bash
git clone <this repository> && cd editjumps
uv sync                 # Runtime dependencies + editable CLI install
uv sync --all-groups    # Complete dependencies including torch/transformers
```

Development dependencies are included in the default sync (`pytest`, `ruff`, `ty`, `pre-commit`, `dvc`). To match CI:

```bash
uv sync --locked --all-extras --dev
```

See [`docs/installation.md`](docs/installation.md) for details on external tools (`hmmscan`, MMseqs2).

## 2. CI Verification Gates

Pull requests must pass the following checks:

| Check | Tool | Command |
|---|---|---|
| Linter | `ruff` | `uv run ruff check .` |
| Type Checker | `ty` | `uv run ty check .` |
| Test Suite | `pytest` | `uv run pytest` |

Local execution:

```bash
make ci              # Runs dependency sync, ruff, ty, and pytest
```

Fast alternatives without dependency sync: `make check` (ruff + ty) and `make test` (pytest).

### Git Hooks

```bash
uv run pre-commit install                        # Runs checks on commit
uv run pre-commit install --hook-type pre-push   # Runs full pytest on push
uv run pre-commit run --all-files                # Manual execution
```

## 3. Code Style

Configured via `[tool.ruff]` in `pyproject.toml`:
- Line length: 120
- Rules: `E` (pycodestyle), `F` (pyflakes), `I` (isort), `D` (pydocstyle), `ANN` (type annotations)
- Python target: 3.10+ compatibility (CI runs on 3.14).

## 4. Tests

Test files mirror source structure (`[tool.pytest.ini_options] testpaths`):

| Target | Test Location |
|---|---|
| Core methods (`editjumps/core/edit_flows/loss.py`) | `editjumps/core/edit_flows/tests/test_loss.py` |
| Shared utilities (`editjumps/core/sequences.py`) | `editjumps/test/test_sequences.py` |
| Pipeline stages (`pipeline/train/evoflows.py`) | `editjumps/test/pipeline/test_evoflows.py` |
| Repository invariants | `editjumps/test/test_repo_*.py` |

Invariants enforced by repository tests:
- Documented `make` targets in code blocks must exist in `Makefile`.
- `docs/claims.md` provenance table must match `editjumps.core.provenance.STAGE_PROVENANCE`.
- DVC pipeline stages must declare valid inputs and outputs.

## 5. Data, Artifacts, and Credentials

- Data and model artifacts are tracked with **DVC**, not git: `make dvc-pull` to fetch,
  `make dvc-push` to publish. The remote is a GCS bucket; `docs/gcp_setup.md` covers access.
- **Never commit** raw data, checkpoints, `.env` files, service-account keys or any credential.
  Large files belong in DVC; secrets belong nowhere in the tree.
- Training runs log to MLflow (`params.yaml`, `mlflow.tracking_uri`). GPU work goes through the
  managed-jobs targets in the `Makefile`, not through a laptop.

## 6. Development Workflow

1. Branch from `main`.
2. Follow standard commit conventions: `type(scope): summary` (`feat(...)`, `fix(...)`, `chore(...)`).
3. Ensure verification checks pass: `make ci`.
4. If modifying pipeline stages, update `dvc.yaml`, `params.yaml`, `editjumps/core/provenance.py`, and run `make provenance`.

## 7. Licensing

The project codebase is released under the [MIT License](LICENSE). Third-party dependencies and vendor tools (such as MMseqs2 under `vendor/`) are governed by their respective upstream licenses.

## 8. Issue Reporting

Submit bug reports and feature requests via GitHub Issues using the provided templates, including reproduction commands and environment details.
