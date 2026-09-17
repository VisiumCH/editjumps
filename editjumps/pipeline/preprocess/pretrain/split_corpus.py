"""Carve a leakage-aware train/validation split out of the OAS pretrain corpus."""

import gzip
from pathlib import Path
from typing import Annotated

import typer

from editjumps.core.cluster_split import cluster_keys, split_by_cluster
from editjumps.core.utils import get_logger

logger = get_logger(__file__)

CORPUS_NAME = "oas_corpus.txt.gz"
CDR_KEYS_NAME = "oas_corpus.cdr_keys.tsv.gz"


def read_lines(path: Path) -> list[str]:
    """Read all newline-stripped lines from a gzipped text file."""
    with gzip.open(path, "rt") as fh:
        return [line.rstrip("\n") for line in fh]


def split_corpus(
    corpus: Path,
    cdr_keys: Path,
    output_folder: Path,
    val_frac: float = 0.05,
    min_seq_id: float = 0.9,
    coverage: float = 0.8,
    mmseqs_mode: str = "easy-cluster",
    chain_handling: str = "combine",
    split_memory_limit: str = "8G",
) -> tuple[Path, Path]:
    """Cluster on CDR-H3+L3 and write leakage-free train/val corpora."""
    if chain_handling != "combine":
        raise NotImplementedError(
            f"chain_handling={chain_handling!r} not implemented; only 'combine' is supported so far. "
            "'separate' (independent heavy/light clustering) is a planned design-space option."
        )
    lines = read_lines(corpus)
    rows = read_lines(cdr_keys)
    if len(lines) != len(rows):
        raise ValueError(
            f"corpus/keys length mismatch: {len(lines)} corpus lines vs {len(rows)} key rows; "
            "rebuild both with `download_oas --pair-chains --emit-cdr-keys`."
        )
    # key = CDR-H3 + CDR-L3 (tab in the sidecar -> direct concatenation here)
    keys = [row.replace("\t", "") for row in rows]
    n_empty = sum(1 for k in keys if not k)
    if n_empty:
        logger.warning(f"{n_empty}/{len(keys)} lines have empty CDR keys (missing in source); grouped by exact key")

    cluster_ids = cluster_keys(
        keys,
        min_seq_id=min_seq_id,
        coverage=coverage,
        mode=mmseqs_mode,
        tmp_dir=output_folder / "mmseqs_tmp",
        split_memory_limit=split_memory_limit,
    )
    assignment = split_by_cluster(cluster_ids, val_frac=val_frac)

    output_folder.mkdir(parents=True, exist_ok=True)
    train_path = output_folder / corpus.name.replace(".txt.gz", ".train.txt.gz")
    val_path = output_folder / corpus.name.replace(".txt.gz", ".val.txt.gz")
    with gzip.open(train_path, "wt") as tr, gzip.open(val_path, "wt") as va:
        for line, side in zip(lines, assignment, strict=True):
            (va if side == "val" else tr).write(line + "\n")

    logger.info(f"Wrote {train_path} and {val_path}")
    return train_path, val_path


def main(
    output_folder: Annotated[Path, typer.Option(help="Folder holding the corpus and where splits are written")] = Path(
        "data/pretrain"
    ),
    val_frac: Annotated[float, typer.Option(help="Target validation fraction (by line count)")] = 0.05,
    min_seq_id: Annotated[float, typer.Option(help="MMseqs2 min identity to cluster two CDR keys")] = 0.9,
    coverage: Annotated[float, typer.Option(help="MMseqs2 min alignment coverage")] = 0.8,
    mmseqs_mode: Annotated[str, typer.Option(help="easy-cluster (sensitive) or easy-linclust (fast)")] = "easy-cluster",
    chain_handling: Annotated[str, typer.Option(help="combine (VH.VL joined) | separate (H/L, planned)")] = "combine",
    corpus_name: Annotated[
        str, typer.Option(help="Corpus filename under --output-folder; the split outputs are named from it")
    ] = CORPUS_NAME,
    cdr_keys_name: Annotated[
        str, typer.Option(help="CDR-keys filename, aligned line-for-line to corpus_name")
    ] = CDR_KEYS_NAME,
    split_memory_limit: Annotated[
        str, typer.Option(help="Cap MMseqs prefilter RAM (e.g. 8G); prevents swapping on small machines")
    ] = "8G",
) -> None:
    """CLI entry point for the leakage-aware corpus split."""
    split_corpus(
        output_folder / corpus_name,
        output_folder / cdr_keys_name,
        output_folder,
        val_frac=val_frac,
        min_seq_id=min_seq_id,
        coverage=coverage,
        mmseqs_mode=mmseqs_mode,
        chain_handling=chain_handling,
        split_memory_limit=split_memory_limit,
    )


if __name__ == "__main__":
    typer.run(main)
