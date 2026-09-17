"""Sequence-level editing API using trained Edit Flows models."""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Annotated

import typer

from editjumps.core.edit_flows.alignment import levenshtein
from editjumps.core.sequences import AA, PAIR_SEP
from editjumps.core.utils import decode_one, encode_one, load_tokenizer

#: Default path searched for restored editor weights.
DEFAULT_MODEL_FOLDER = Path("data/pretrain/edit_flows_restored")

#: Required artifacts in a model folder.
REQUIRED_ARTIFACTS: tuple[str, str] = ("encoder", "evoflows_model.pt")

#: Default run tag for checkpoint restoration.
DEFAULT_RUN_TAG = "faithful-appendixa"

#: Private bucket for model checkpoints.
CHECKPOINT_BUCKET = os.environ.get("DVC_BUCKET", "gs://<DVC_BUCKET>")

#: Head parameterisations matching DEFAULT_RUN_TAG.
DEFAULT_RATE_HEAD = "mlp"
DEFAULT_Q_HEAD = "esm_lm_head"

#: Baseline mean per-position rate for translating edits to clock.
DEFAULT_LAMBDA_BAR = 0.0990

#: Empirical power-law exponent relating clock to edits.
CALIBRATION_EXPONENT = 0.762

#: Sample sizes for empirical rate calibration.
CALIBRATION_PROBE_N = 4
CALIBRATION_MIN_N = 4

#: Empirical range of mean per-position rates.
LAMBDA_BAR_RANGE: tuple[float, float] = (0.0933, 0.2331)

#: Default step resolution for the Euler sampler.
DEFAULT_N_STEPS = 50


class CheckpointNotFoundError(FileNotFoundError):
    """No usable editor checkpoint was found, and we refuse to invent one."""


@dataclass(frozen=True)
class Variant:
    """One generated sequence and how far it actually moved."""

    sequence: str
    edit_distance: int


@dataclass(frozen=True)
class EditReport:
    """Report summarizing sequence editing execution and parameters."""

    sequence: str
    variants: list[Variant]
    model: str
    rate_head: str
    q_head: str
    clock: float
    requested_edits: int | None
    lambda_bar: float
    sampler: str
    seed: int
    source_id: str | None
    calibrated: bool = False
    probe_n: int = 0

    def as_dict(self) -> dict[str, object]:
        """Render as JSON-serialisable plain data."""
        return {
            "sequence": self.sequence,
            "n_variants": len(self.variants),
            "variants": [
                {"sequence": v.sequence, "edit_distance": v.edit_distance} for v in self.variants
            ],
            "edit_distance_mean": (
                sum(v.edit_distance for v in self.variants) / len(self.variants)
                if self.variants
                else 0.0
            ),
            "model": self.model,
            "rate_head": self.rate_head,
            "q_head": self.q_head,
            "clock": self.clock,
            "calibrated": self.calibrated,
            "probe_n": self.probe_n,
            "requested_edits": self.requested_edits,
            "lambda_bar": self.lambda_bar,
            "sampler": self.sampler,
            "seed": self.seed,
            "source_id": self.source_id,
            # Stated in the payload, not just the docs, because the absence of a score is the
            # kind of thing a consumer would otherwise assume was an omission to be filled in.
            "quality_score": None,
            "quality_score_note": (
                "No quality/confidence score is available: every distributional metric here "
                "needs a holdout from the input's own protein family, which a single sequence "
                "does not provide. See `editjumps generation-eval --family-fasta`."
            ),
        }


def clock_for_edits(edits: int, lambda_bar: float = DEFAULT_LAMBDA_BAR) -> float:
    """Translate a requested edit count into the clock that should roughly deliver it."""
    if edits <= 0:
        raise ValueError(f"edits must be positive, got {edits}")
    if lambda_bar <= 0:
        raise ValueError(f"lambda_bar must be positive, got {lambda_bar}")
    return edits / lambda_bar


def missing_checkpoint_message(searched: Path) -> str:
    """Build the no-weights error: what is missing, and the exact command that fixes it."""
    return (
        f"No editor checkpoint at {searched}.\n"
        f"\n"
        f"`edit` needs trained weights and will not emit random sequences instead, so there is "
        f"nothing sensible for it to do until you point it at some.\n"
        f"\n"
        f"If you have access to the project's GCP bucket, restore the reported arm and retry:\n"
        f"\n"
        f"    make restore-editor \\\n"
        f"      CKPT={CHECKPOINT_BUCKET}/checkpoints/experiments/{DEFAULT_RUN_TAG}/checkpoint.pt \\\n"
        f"      MODEL_NAME=facebook/esm2_t12_35M_UR50D \\\n"
        f"      RATE_HEAD={DEFAULT_RATE_HEAD} Q_HEAD={DEFAULT_Q_HEAD}\n"
        f"\n"
        f"or equivalently:\n"
        f"\n"
        f"    uv run --group train editjumps restore-editor \\\n"
        f"      --checkpoint {CHECKPOINT_BUCKET}/checkpoints/experiments/{DEFAULT_RUN_TAG}/checkpoint.pt \\\n"
        f"      --output-folder {DEFAULT_MODEL_FOLDER} \\\n"
        f"      --model-name facebook/esm2_t12_35M_UR50D \\\n"
        f"      --rate-head {DEFAULT_RATE_HEAD} --q-head {DEFAULT_Q_HEAD}\n"
        f"\n"
        f"That bucket is PRIVATE to the $PROJECT_ID project. There is no public mirror "
        f"and no download will be attempted on your behalf: outside the org the restore above "
        f"fails with AccessDenied. Your options are then to train your own editor "
        f"(`editjumps train-edit-flows`) or to obtain a checkpoint from the authors, and pass "
        f"either with --model / model=.\n"
        f"\n"
        f"A model folder is one holding {REQUIRED_ARTIFACTS[0]}/ and {REQUIRED_ARTIFACTS[1]}. "
        f"Whatever you pass, its --rate-head/--q-head must match how it was trained "
        f"(default here: {DEFAULT_RATE_HEAD}/{DEFAULT_Q_HEAD}, i.e. {DEFAULT_RUN_TAG})."
    )


def resolve_model_folder(model: str | Path | None = None) -> Path:
    """Find a usable model folder, or raise the message that says how to get one."""
    folder = Path(model) if model is not None else DEFAULT_MODEL_FOLDER
    # Check the artifacts, not just the directory: a half-written restore leaves the folder
    # there, and `load_trained` would fail deep inside torch with a much worse message.
    if all((folder / name).exists() for name in REQUIRED_ARTIFACTS):
        return folder
    raise CheckpointNotFoundError(missing_checkpoint_message(folder))


def read_sequence(source: str | Path) -> tuple[str, str | None]:
    """Resolve a raw sequence string or a FASTA path into one sequence."""
    text = str(source).strip()
    source_id: str | None = None

    # A FASTA path, not a sequence: decided by the file existing, so a sequence is never
    # mistaken for a filename (and a typo'd path is never silently read as a sequence).
    if text and not text.startswith(">") and Path(text).is_file():
        from editjumps.pipeline.preprocess.pretrain.seed_homologs import read_fasta

        records = read_fasta(Path(text))
        if len(records) != 1:
            raise ValueError(
                f"{text}: expected exactly one FASTA record, found {len(records)}. "
                f"`edit` edits one sequence. Split the file, or - for a VH/VL pair - pass the "
                f"single joined string 'VH{PAIR_SEP}VL', which is the form the editor was "
                f"trained on."
            )
        source_id, text = next(iter(records.items()))

    sequence = text.strip().upper().replace(" ", "").replace("\n", "")
    if not sequence:
        raise ValueError("empty sequence")
    # PAIR_SEP is legal: it is the trained-on VH.VL join and a real ESM-2 vocabulary token.
    illegal = sorted(set(sequence) - AA - {PAIR_SEP})
    if illegal:
        raise ValueError(
            f"not an amino-acid sequence: unexpected character(s) {''.join(illegal)!r}. "
            f"Expected the 20 standard amino acids (and {PAIR_SEP!r} to join VH to VL), or a "
            f"path to a FASTA file."
        )
    return sequence, source_id


def edit_report(
    sequence: str | Path,
    n: int = 10,
    edits: int | None = 5,
    *,
    clock: float | None = None,
    model: str | Path | None = None,
    rate_head: str = DEFAULT_RATE_HEAD,
    q_head: str = DEFAULT_Q_HEAD,
    sampler: str = "euler",
    n_steps: int = DEFAULT_N_STEPS,
    seed: int = 0,
    lambda_bar: float = DEFAULT_LAMBDA_BAR,
) -> EditReport:
    """Generate variants and return them with the settings that produced them. `edit` is the friendly."""
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")

    text, source_id = read_sequence(sequence)
    folder = resolve_model_folder(model)

    if clock is not None:
        resolved_clock, requested = float(clock), None
    elif edits is not None:
        resolved_clock, requested = clock_for_edits(edits, lambda_bar), edits
    else:
        raise ValueError("pass either edits= (an approximate budget) or clock= (the mechanism)")

    from editjumps.core.edit_flows.stages import resolve_sampler
    from editjumps.pipeline.train.evoflows import EvoFlowsModel, pick_device

    sample = resolve_sampler(sampler)
    try:
        loaded = EvoFlowsModel.load_trained(folder, rate_head=rate_head, q_head=q_head).to(pick_device())
    except RuntimeError as exc:
        raise CheckpointNotFoundError(
            f"{folder} did not load with rate_head={rate_head!r}, q_head={q_head!r}.\n"
            f"Parameters must match training configuration. {DEFAULT_RUN_TAG} uses "
            f"rate_head={DEFAULT_RATE_HEAD!r}, q_head={DEFAULT_Q_HEAD!r}.\n"
            f"\nUnderlying error: {exc}"
        ) from exc
    loaded.eval()
    tokenizer = load_tokenizer(folder / "encoder")
    input_ids = encode_one(tokenizer, text)

    def draw(clock_value: float, count: int, offset: int) -> list[Variant]:
        """Sample variants at a given clock."""
        drawn = []
        for index in range(count):
            out = sample(
                loaded,
                list(input_ids),
                n_steps=n_steps,
                clock=clock_value,
                rng=random.Random(seed + offset + index),
            )
            decoded = decode_one(tokenizer, out)
            if PAIR_SEP not in text:
                decoded = decoded.replace(PAIR_SEP, "")
            drawn.append(Variant(sequence=decoded, edit_distance=levenshtein(text, decoded)))
        return drawn

    calibrated, probe_n = False, 0
    if requested is not None and n >= CALIBRATION_MIN_N:
        probe_n = min(CALIBRATION_PROBE_N, max(2, n // 3))
        probe = draw(resolved_clock, probe_n, 0)
        calibrated = True
        realised = mean(variant.edit_distance for variant in probe)
        if realised > 0:
            corrected = resolved_clock * (requested / realised) ** (1.0 / CALIBRATION_EXPONENT)
            resolved_clock = float(min(max(corrected, resolved_clock / 8.0), resolved_clock * 8.0))

    variants: list[Variant] = draw(resolved_clock, n, 1000)

    return EditReport(
        sequence=text,
        variants=variants,
        model=str(folder),
        rate_head=rate_head,
        q_head=q_head,
        clock=resolved_clock,
        requested_edits=requested,
        lambda_bar=lambda_bar,
        calibrated=calibrated,
        probe_n=probe_n,
        sampler=sampler,
        seed=seed,
        source_id=source_id,
    )


def edit(
    sequence: str | Path,
    n: int = 10,
    edits: int | None = 5,
    *,
    clock: float | None = None,
    model: str | Path | None = None,
    rate_head: str = DEFAULT_RATE_HEAD,
    q_head: str = DEFAULT_Q_HEAD,
    sampler: str = "euler",
    n_steps: int = DEFAULT_N_STEPS,
    seed: int = 0,
    lambda_bar: float = DEFAULT_LAMBDA_BAR,
) -> list[Variant]:
    """Generate variants with approximate edit counts (requires holdout for scoring)."""
    return edit_report(
        sequence,
        n=n,
        edits=edits,
        clock=clock,
        model=model,
        rate_head=rate_head,
        q_head=q_head,
        sampler=sampler,
        n_steps=n_steps,
        seed=seed,
        lambda_bar=lambda_bar,
    ).variants


def main(
    sequence: Annotated[str, typer.Option(help="Amino-acid sequence, or a path to a single-record FASTA")] = "",
    n: Annotated[int, typer.Option(help="How many variants to generate")] = 10,
    edits: Annotated[int, typer.Option(help="Approximate mean edits per variant (translated to a clock)")] = 5,
    clock: Annotated[
        float, typer.Option(help="Clock normalisation, straight to the sampler; overrides --edits. 0 = unset")
    ] = 0.0,
    model: Annotated[str, typer.Option(help=f"Model folder; default {DEFAULT_MODEL_FOLDER}")] = "",
    rate_head: Annotated[str, typer.Option(help="linear | mlp; must match the checkpoint")] = DEFAULT_RATE_HEAD,
    q_head: Annotated[str, typer.Option(help="fresh | esm_lm_head; must match the checkpoint")] = DEFAULT_Q_HEAD,
    sampler: Annotated[str, typer.Option(
        help="euler (fixed grid) | gillespie (no grid, rate frozen between events); "
             "neither is exact")] = "euler",
    n_steps: Annotated[int, typer.Option(help="Euler grid resolution (ignored by gillespie)")] = DEFAULT_N_STEPS,
    seed: Annotated[int, typer.Option(help="Base RNG seed; variant i uses seed + i")] = 0,
    lambda_bar: Annotated[
        float, typer.Option(help="Mean per-position rate used for --edits -> clock")
    ] = DEFAULT_LAMBDA_BAR,
    as_json: Annotated[bool, typer.Option("--json", help="Emit machine-readable JSON on stdout")] = False,
) -> None:
    """Edit sequence with approximate edit counts (requires holdout for scoring)."""
    if not sequence:
        typer.secho("--sequence is required (an amino-acid string, or a FASTA path)", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)
    try:
        report = edit_report(
            sequence, n=n, edits=edits, clock=clock or None, model=model or None, rate_head=rate_head,
            q_head=q_head, sampler=sampler, n_steps=n_steps, seed=seed, lambda_bar=lambda_bar,
        )
    # Both are user-fixable: a missing checkpoint or a malformed sequence. Their messages already
    # say what to do, so print them and exit - a traceback here only buries the instruction.
    except (CheckpointNotFoundError, ValueError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None

    if as_json:
        typer.echo(json.dumps(report.as_dict(), indent=2))
        return

    distances = [v.edit_distance for v in report.variants]
    typer.echo(f"input   {len(report.sequence)} aa" + (f"  ({report.source_id})" if report.source_id else ""))
    typer.echo(
        f"model   {report.model}  (rate_head={report.rate_head}, q_head={report.q_head}, "
        f"sampler={report.sampler})"
    )
    # Say where the clock came from, because the two provenances are not interchangeable: a calibrated clock.
    if report.requested_edits is None:
        budget = f"clock {report.clock:.1f} (given)"
    elif report.calibrated:
        budget = (
            f"clock {report.clock:.1f} (calibrated to --edits {report.requested_edits} on a "
            f"{report.probe_n}-variant probe; still approximate)"
        )
    else:
        budget = (
            f"clock {report.clock:.1f} (from --edits {report.requested_edits} via lambda_bar "
            f"{report.lambda_bar:g}, uncalibrated; approximate to a factor of ~3)"
        )
    typer.echo(f"budget  {budget}")
    typer.echo(f"edits   realised mean {sum(distances) / len(distances):.1f}, range {min(distances)}-{max(distances)}")
    typer.echo("")
    for index, variant in enumerate(report.variants):
        typer.echo(f"{index:>3}  {variant.edit_distance:>3} edits  {variant.sequence}")
    typer.echo("")
    # Repeated at the point of use, not just in --help: this is where someone is looking at a
    # list of sequences and deciding which to order.
    typer.echo(
        "No quality score is available for these: scoring needs a holdout from the input's own "
        "protein family. Edit distance is not confidence."
    )
