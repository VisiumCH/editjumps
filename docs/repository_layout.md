# Repository Layout

Organization of directories and code modules:

- **Method implementations:** Subpackages under `editjumps/core/` (`edit_flows/`, `evodiff_msa/`, `evotune/`) contain method-specific logic, tests, and documentation.
- **Isolated environments:** `evodiff_msa_runner.py` executes in an isolated environment (`.evodiff_env`, requiring `numpy<2`) and is invoked as a subprocess rather than imported directly.
- **Evaluation metrics:** Subdirectories under `metrics/` contain evaluation results and local documentation on comparability.

```
editjumps/editing.py       `editjumps edit` / `from editjumps import edit` — the product surface
editjumps/main.py          the `editjumps` CLI: every stage as a subcommand, grouped by panel
editjumps/ranking.py       `editjumps rank` — score and order candidate sequences

editjumps/core/            method-independent maths and plumbing: IMGT numbering (cdr.py), the
                         family and cluster splits, the Appendix-B metrics
                         (generation_metrics.py), the three-file run config, the claims map
                         (provenance.py), the shared writer that stamps a run's git commit onto
                         its metrics (utils.py)
  edit_flows/            EvoFlows' editor — Edit Flows underneath. Its own README + tests/
  evodiff_msa/           EvoDiff-MSA, §4.2's fifth baseline
  evotune/               §2.2 evotuning, §4.2's two evotuned-PLM baselines

editjumps/pipeline/
  preprocess/            OAS corpus, the corpus split, the training pairs, the seed families
  train/                 editor training, the trunk pretrain, evotuning, throughput calibration
  evaluate/              the §4.2/§4.3 evaluation, every baseline, the consolidator, the paired
                         bootstrap, the inference trace
editjumps/measurements/   standalone measurement scripts and sweeps cited in docs/findings.md
editjumps/repo/            repo maintenance that CI and `make` DO depend on: the stage-coverage
                         audit, the `dvc pull` dep resolver a cloud job calls by path, the README
                         image generators, the MLflow-UI launcher.
editjumps/test/            the shared-core, CLI, repo-invariant and artefact suites; its
                         `pipeline/` subdirectory covers the DAG stages. A method's own tests
                         live beside the method, not here.

deploy/gcp/              SkyPilot job specs + Terraform for the GPU box and MLflow server
docs/                    everything below
metrics/                 every committed result. `disjoint/` is the authoritative §4.2 table;
                         `all_results.{json,csv}` is a generated view over the per-run files
.github/images/          the README hero SVGs (light + dark); `make hero` regenerates both
vendor/                  MMseqs2's licence and a note on where its binary comes from; no source

params.yaml              every tunable knob the PIPELINE reads
dvc.yaml                 the DAG
Makefile                 every workflow, one line each (`make help`)
```

## Where to go next

| | |
|---|---|
| [`installation.md`](installation.md) | from a clone to an environment that can run this |
| [`reproducing.md`](reproducing.md) | regenerating the reported numbers, and the comparability rules |
| [`claims.md`](claims.md) | which stage belongs to which research claim |
