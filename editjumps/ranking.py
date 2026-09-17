"""Rank candidate sequences by ESM-2 naturalness."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import typer

#: Scoring model.
DEFAULT_SCORER = "facebook/esm2_t33_650M_UR50D"

#: Positions to score per sequence. ``None`` scores every position, one forward pass each, which on CPU is.
DEFAULT_MAX_POSITIONS = 40


@dataclass(frozen=True)
class Ranked:
    """One scored candidate."""

    sequence: str
    score: float
    rank: int
    index: int


def rank(
    sequences: list[str] | Path | str,
    *,
    scorer: str = DEFAULT_SCORER,
    max_positions: int | None = DEFAULT_MAX_POSITIONS,
    seed: int = 0,
) -> list[Ranked]:
    """Score candidates by ESM-2 naturalness and return them most-natural first."""
    from editjumps.core.pseudo_likelihood import pseudo_log_likelihood

    # Resolve the input to a plain list of strings first, so nothing below has to wonder.
    candidates: list[str]
    if isinstance(sequences, Path):
        if not sequences.exists():
            raise ValueError(f"{sequences} does not exist")
        candidates = read_fasta(sequences)
    elif isinstance(sequences, str):
        as_path = Path(sequences)
        candidates = read_fasta(as_path) if as_path.exists() else [sequences]
    else:
        candidates = list(sequences)

    if not candidates:
        return []
    blank = [i for i, s in enumerate(candidates) if not s.strip()]
    if blank:
        raise ValueError(f"empty candidate(s) at index {blank}; nothing to score")

    cleaned = [s.strip().upper().replace(" ", "") for s in candidates]
    scored = pseudo_log_likelihood(cleaned, model_name=scorer, seed=seed,
                                   max_positions=max_positions)
    order = sorted(range(len(cleaned)), key=lambda i: -scored["per_sequence"][i])
    return [Ranked(sequence=cleaned[i], score=scored["per_sequence"][i], rank=place + 1, index=i)
            for place, i in enumerate(order)]


def read_fasta(path: Path) -> list[str]:
    """Read every record of a FASTA into a list of sequences."""
    records, current = [], []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if current:
                records.append("".join(current))
            current = []
        elif line.strip():
            current.append(line.strip())
    if current:
        records.append("".join(current))
    return records


def main(
    sequences: "list[str]" = typer.Option([], "--sequence", help="A candidate; repeat for more"),
    fasta: str = typer.Option("", help="FASTA of candidates, instead of --sequence"),
    scorer: str = typer.Option(DEFAULT_SCORER, help="Scoring model (a stock ESM-2)"),
    max_positions: int = typer.Option(DEFAULT_MAX_POSITIONS,
                                      help="Positions scored per sequence; 0 scores all"),
    seed: int = typer.Option(0, help="Seed for B.2's random masking order"),
    as_json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON on stdout"),
) -> None:
    r"""Rank candidate sequences by ESM-2 naturalness, most natural first. \b editjumps rank --sequence."""
    import json

    candidates: list[str] | Path
    if sequences:
        candidates = list(sequences)
    elif fasta:
        candidates = Path(fasta)
    else:
        typer.secho("pass --sequence (repeatable) or --fasta", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)

    ranked = rank(candidates, scorer=scorer,
                  max_positions=None if max_positions == 0 else max_positions, seed=seed)
    if as_json:
        print(json.dumps({
            "scorer": scorer,
            "max_positions": None if max_positions == 0 else max_positions,
            "seed": seed,
            "metric": "esm2_pseudo_log_likelihood_mean_per_position",
            "ranked": [{"rank": r.rank, "index": r.index, "score": r.score,
                        "sequence": r.sequence} for r in ranked],
        }, indent=2))
        return

    positions = "all" if max_positions == 0 else str(max_positions)
    print(f"scorer  {scorer}")
    print(f"metric  ESM-2 pseudo-log-likelihood, mean per position over {positions} positions "
          f"(higher = more natural)")
    print(f"ranked  {len(ranked)} candidates\n")
    for r in ranked:
        print(f"  {r.rank:>2}  {r.score:+.4f}  (input #{r.index})  {r.sequence}")
    print("\nThis ranks naturalness under a stock protein LM, not your property. Nothing here has "
          "been told\nwhat any downstream property is.")


if __name__ == "__main__":
    typer.run(main)
