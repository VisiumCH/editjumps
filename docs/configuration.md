# Configuration

`params.yaml` is the single source of truth for pipeline execution and evaluation settings.

## Conceptual Configuration Split

Rather than maintaining duplicated configuration files, [`editjumps/core/run_config.py`](../editjumps/core/run_config.py) groups the parameters from `params.yaml` into three conceptual domains:

| Group | Defined By | Holds | May two runs differ? |
|---|---|---|---|
| **Target** | `TargetConfig` | Which seed family, the edit budget/clock (`edit_flows.clock`, `evotune.family`) | **Yes** — that difference *is* the experiment |
| **Comparability** | `ComparabilityConfig` | Holdout size, alignment frame, MMD sample sizes, ceiling size, matched mutation budget | **No** — if these differ, the numbers are not comparable |
| **Arm** | `ArmConfig` | Checkpoint, rate/Q head, schedule, sampler, and baseline generator settings | **Yes** — that is what is under test |

The configuration split isolates experimental variables from evaluation controls:
- **Target**: Defines dataset and target sequence settings.
- **Comparability**: Pins evaluation controls (holdout size, alignment frame, sample sizes) that must remain identical across runs for valid comparison. Two runs should only be compared side-by-side if their evaluation shapes match.
- **Arm**: Defines model checkpoint, head parameterizations, sampler, and baseline generator settings.

`editjumps evaluate` calculates a `comparability_fingerprint` hash in `metrics/standard/run_config.json` to verify that compared runs share matching evaluation parameters. See [`editjumps/core/run_config.py`](../editjumps/core/run_config.py) for field definitions.

## Running Evaluations

By default, running:

```bash
uv run editjumps evaluate
```

reads the configuration directly from `params.yaml`.

### Testing Multiple Arms

To evaluate an alternate model checkpoint or head parameterization without modifying `params.yaml`:

1. **Option A: Write standalone YAML files**:
   ```bash
   uv run editjumps evaluate --init --config-dir custom_configs
   # edit custom_configs/arm.yaml, then:
   uv run editjumps evaluate --config-dir custom_configs --out-dir metrics/arm_B
   ```

2. **Option B: Pass an override arm file directly**:
   ```bash
   uv run editjumps evaluate --arm custom_arm.yaml --out-dir metrics/arm_B
   ```
