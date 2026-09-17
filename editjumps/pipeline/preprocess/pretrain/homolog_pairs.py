"""Build homolog sequence pairs for Edit Flows / EvoFlows training (Phase 2a)."""

from __future__ import annotations

import gzip
import random
from collections.abc import Sequence
from itertools import combinations
from pathlib import Path
from typing import Annotated

import typer

from editjumps.core.cluster_split import cluster_keys
from editjumps.core.utils import get_logger
from editjumps.pipeline.preprocess.pretrain.split_corpus import CDR_KEYS_NAME, CORPUS_NAME, read_lines

logger = get_logger(__file__)

OUTPUT_NAME = "oas_homolog_pairs.tsv.gz"


def group_families(cluster_ids: Sequence[int], lines: Sequence[str]) -> dict[int, list[str]]:
    """Group corpus lines into homolog families by cluster id, de-duplicating within each."""
    families: dict[int, list[str]] = {}
    seen: dict[int, set[str]] = {}
    for cid, line in zip(cluster_ids, lines, strict=True):
        members = families.setdefault(cid, [])
        seen_in = seen.setdefault(cid, set())
        if line not in seen_in:  # homologs, not exact dupes
            seen_in.add(line)
            members.append(line)
    return families


def sample_pairs(
    families: dict[int, list[str]], max_pairs_per_family: int, rng: random.Random
) -> list[tuple[str, str]]:
    """Emit unordered intra-family pairs, capped per family to bound the dataset. §4.2 enumerates *all*."""
    pairs: list[tuple[str, str]] = []
    for members in families.values():
        k = len(members)
        if k < 2:
            continue
        if k * (k - 1) // 2 <= max_pairs_per_family:
            pairs.extend((members[a], members[b]) for a, b in combinations(range(k), 2))
        else:
            seen: set[tuple[int, int]] = set()
            while len(seen) < max_pairs_per_family:
                i, j = sorted(rng.sample(range(k), 2))
                if (i, j) not in seen:
                    seen.add((i, j))
                    pairs.append((members[i], members[j]))
    return pairs


def build_homolog_pairs(
    output_folder: Path,
    min_seq_id: float = 0.5,
    coverage: float = 0.8,
    mmseqs_mode: str = "easy-cluster",
    max_pairs_per_family: int = 20,
    max_lines: int = 0,
    seed: int = 0,
    direction: str = "none",
    corpus_name: str = CORPUS_NAME,
    cdr_keys_name: str = CDR_KEYS_NAME,
    split_memory_limit: str = "8G",
) -> Path:
    """Cluster the OAS corpus into homolog families and write intra-family pairs."""
    # Validated BEFORE the corpus read and the MMseqs run: `improving` used to be refused after
    # clustering the whole corpus, so a user paid an hours-long sweep to be told the option is gone.
    if direction == "improving":
        raise ValueError(
            "homolog_pairs.direction='improving' orients each pair so x1 is the better member of a "
            "target property. That needs a property registry, which this repository does not ship: "
            "EvoFlows trains on unoriented homolog pairs and `none` is the faithful setting. See "
            "docs/claims.md, 'Why there is no target property'."
        )
    if direction != "none":
        raise ValueError(f"unknown direction {direction!r}; use 'none' or 'improving'")

    lines = read_lines(output_folder / corpus_name)
    rows = read_lines(output_folder / cdr_keys_name)
    if len(lines) != len(rows):
        raise ValueError(f"corpus/keys length mismatch: {len(lines)} vs {len(rows)}; rebuild with download_oas.")
    if max_lines:
        lines, rows = lines[:max_lines], rows[:max_lines]

    keys = [row.replace("\t", "") for row in rows]
    cluster_ids = cluster_keys(
        keys, min_seq_id=min_seq_id, coverage=coverage, mode=mmseqs_mode,
        tmp_dir=output_folder / "mmseqs_tmp_pairs", split_memory_limit=split_memory_limit,
        stage="build_homolog_pairs",
    )
    families = group_families(cluster_ids, lines)
    n_multi = sum(1 for m in families.values() if len(m) >= 2)
    pairs = sample_pairs(families, max_pairs_per_family=max_pairs_per_family, rng=random.Random(seed))
    logger.info(f"{len(families)} families ({n_multi} with >=2 members) -> {len(pairs)} homolog pairs")

    output_folder.mkdir(parents=True, exist_ok=True)
    output_path = output_folder / OUTPUT_NAME
    with gzip.open(output_path, "wt") as fh:
        for x0, x1 in pairs:
            fh.write(f"{x0}\t{x1}\n")
    logger.info(f"Wrote {output_path}")
    return output_path


def main(
    output_folder: Annotated[Path, typer.Option()] = Path("data/pretrain"),
    min_seq_id: Annotated[float, typer.Option(help="MMseqs2 min identity for homolog families")] = 0.5,
    coverage: Annotated[float, typer.Option(help="MMseqs2 min alignment coverage")] = 0.8,
    mmseqs_mode: Annotated[str, typer.Option(help="easy-cluster (sensitive) or easy-linclust (fast)")] = "easy-cluster",
    max_pairs_per_family: Annotated[int, typer.Option(help="Cap on pairs emitted per homolog family")] = 20,
    max_lines: Annotated[int, typer.Option(help="Use only the first N corpus lines (0 = all)")] = 0,
    seed: Annotated[int, typer.Option(help="RNG seed for pair sampling")] = 0,
    direction: Annotated[
        str, typer.Option(help="none (symmetric, EvoFlows-faithful); improving needs a property registry")
    ] = "none",
    corpus_name: Annotated[str, typer.Option(help="Corpus filename under --output-folder")] = CORPUS_NAME,
    cdr_keys_name: Annotated[str, typer.Option(help="CDR-keys filename, aligned to corpus_name")] = CDR_KEYS_NAME,
    split_memory_limit: Annotated[
        str, typer.Option(help="Cap MMseqs prefilter RAM (e.g. 8G); prevents swapping on small machines")
    ] = "8G",
) -> None:
    """CLI entry point: build homolog pairs for Edit Flows training."""
    build_homolog_pairs(
        output_folder,
        min_seq_id=min_seq_id,
        coverage=coverage,
        mmseqs_mode=mmseqs_mode,
        max_pairs_per_family=max_pairs_per_family,
        max_lines=max_lines,
        seed=seed,
        direction=direction,
        corpus_name=corpus_name,
        cdr_keys_name=cdr_keys_name,
        split_memory_limit=split_memory_limit,
    )


if __name__ == "__main__":
    typer.run(main)
