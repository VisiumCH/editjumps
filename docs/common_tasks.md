# Common Tasks

Frequently used `make` targets (run `make help` for all targets):

| Task | Command |
|---|---|
| Lint, type-check, test | `make check` · `make test` |
| Rebuild `pretrain_esm` and its deps | `make repro` |
| Rebuild the whole DAG | `make repro STAGES=` |
| Restore a loadable editor from a checkpoint | `make restore-editor` |
| Train the editor on a GPU | `make jobs-train` (managed spot, auto-resumes) |
| Run §4.2's EvoDiff-MSA baseline | `make install-evodiff` then `make evodiff-baseline` |

One third-party model lives in an isolated environment because its dependencies conflict with
ours: `make install-evodiff`. It downloads ~380 MB of MSA weights on its first run, and needs no
system binaries.
