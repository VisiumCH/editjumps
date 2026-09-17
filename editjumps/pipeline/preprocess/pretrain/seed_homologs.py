"""Build EvoFlows-style homolog families from seed sequences, and report them in the paper's format."""

import gzip
import shutil
import statistics
import subprocess
import tempfile
from pathlib import Path
from typing import Annotated

import typer

from editjumps.core.utils import get_logger, write_metrics

logger = get_logger(__file__)

CHAIN_CHOICES = ("heavy", "light", "joined")


def read_fasta(path: Path) -> dict[str, str]:
    """Read a FASTA file into ``{id: sequence}``."""
    entries, name, chunks = {}, None, []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if name:
                entries[name] = "".join(chunks)
            name, chunks = line[1:].split()[0], []
        elif line.strip():
            chunks.append(line.strip())
    if name:
        entries[name] = "".join(chunks)
    return entries


def corpus_to_fasta(corpus: Path, target: Path, chain: str, sep: str = ".") -> int:
    """Write the corpus as FASTA, taking one chain per antibody."""
    if chain not in CHAIN_CHOICES:
        raise ValueError(f"chain={chain!r}; options: {CHAIN_CHOICES}")
    written = 0
    with gzip.open(corpus, "rt") as inp, open(target, "w") as out:
        for i, line in enumerate(inp):
            joined = line.rstrip("\n")
            if chain == "joined":
                seq = joined.replace(sep, "")
            else:
                parts = joined.split(sep)
                seq = parts[0] if chain == "heavy" else (parts[1] if len(parts) > 1 else "")
            if len(seq) > 50:
                out.write(f">{i}\n{seq}\n")
                written += 1
    return written


def summarise(hits: list[dict], seed_length: int) -> dict:
    """Reduce one seed's hits to the paper's Table 2 columns."""
    def stat(key: str) -> tuple[float, float]:
        values = [h[key] for h in hits]
        if not values:
            return (0.0, 0.0)
        return (statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0)

    length_mean, length_sd = stat("length")
    qcov_mean, qcov_sd = stat("qcov")
    tcov_mean, tcov_sd = stat("tcov")
    return {
        "n_homologs": len(hits),
        "seed_length": seed_length,
        "length_mean": round(length_mean, 1), "length_sd": round(length_sd, 1),
        "query_coverage_mean": round(qcov_mean, 2), "query_coverage_sd": round(qcov_sd, 2),
        "target_coverage_mean": round(tcov_mean, 2), "target_coverage_sd": round(tcov_sd, 2),
    }


def main(
    seeds: Annotated[Path, typer.Option(help="FASTA of seed sequences (one family per seed)")] = Path(
        # In the package next to this stage, not the gitignored data/**: SkyPilot's workdir sync
        # respects .gitignore, so a seed file there never reaches the VM that runs the search.
        "editjumps/pipeline/preprocess/pretrain/seeds/evoflows_seeds.fasta"
    ),
    corpus: Annotated[Path, typer.Option()] = Path("data/pretrain/oas_corpus.txt.gz"),
    chain: Annotated[str, typer.Option(help="heavy | light | joined - which chain to search")] = "heavy",
    evalue: Annotated[float, typer.Option(help="Max E-value; the paper uses 1e-1")] = 0.1,
    coverage: Annotated[float, typer.Option(help="Min query coverage; the paper uses 0.8")] = 0.8,
    sensitivity: Annotated[float, typer.Option(help="MMseqs -s; higher finds remoter homologs")] = 7.5,
    min_identity: Annotated[
        float, typer.Option(help="Keep hits at >= this identity to the seed; 0 = the paper's E-value only")
    ] = 0.0,
    max_seqs: Annotated[
        int, typer.Option(help="Prefilter hits per query; MMseqs' default of 300 silently caps families")
    ] = 100000,
    split_memory_limit: Annotated[str, typer.Option(help="Cap MMseqs prefilter RAM")] = "4G",
    output_folder: Annotated[Path, typer.Option()] = Path("data/interim/seed_families"),
    metrics_path: Annotated[Path, typer.Option()] = Path("metrics/seed_homologs.json"),
) -> None:
    """Search each seed against the corpus and report family statistics in Table 2's columns."""
    if shutil.which("mmseqs") is None:
        raise RuntimeError("mmseqs not found on PATH - run `bash editjumps/core/install_mmseqs.sh`")

    seed_seqs = read_fasta(seeds)
    logger.info(f"{len(seed_seqs)} seeds: {', '.join(seed_seqs)}")
    output_folder.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        target_fasta, hits_path = tmp_dir / "corpus.fasta", tmp_dir / "hits.tsv"
        n_targets = corpus_to_fasta(corpus, target_fasta, chain)
        logger.info(f"searching {len(seed_seqs)} seeds against {n_targets:,} {chain} chains "
                    f"(E-value <= {evalue}, query coverage >= {coverage}, -s {sensitivity})")
        subprocess.run(
            ["mmseqs", "easy-search", str(seeds), str(target_fasta), str(hits_path), str(tmp_dir / "mmseqs_tmp"),
             "-e", str(evalue), "-c", str(coverage), "--cov-mode", "0", "-s", str(sensitivity),
             # WITHOUT this, MMseqs keeps only its default 300 prefilter hits per query, so the "family size".
             "--max-seqs", str(max_seqs),
             "--format-output", "query,target,fident,alnlen,evalue,qcov,tcov,tseq", "-v", "1",
             "--split-memory-limit", split_memory_limit],
            check=True, capture_output=True, text=True,
        )

        families: dict[str, list[dict]] = {name: [] for name in seed_seqs}
        if hits_path.exists():
            for row in hits_path.read_text().splitlines():
                parts = row.split("\t")
                if len(parts) < 8:
                    continue
                query, _target, fident, _alnlen, _evalue, qcov, tcov, tseq = parts[:8]
                # fident is what makes an antibody family meaningful: E-value <= 0.1 admits every V
                # domain (shared framework), so identity is the filter that selects relatives.
                if query in families and float(fident) >= min_identity:
                    families[query].append({"length": len(tseq), "qcov": float(qcov),
                                            "tcov": float(tcov), "fident": float(fident),
                                            "seq": tseq})

    report = {}
    for name, hits in families.items():
        report[name] = summarise(hits, len(seed_seqs[name]))
        family_path = output_folder / f"{name}.fasta"
        with open(family_path, "w") as out:
            for i, hit in enumerate(hits):
                out.write(f">{name}_h{i}\n{hit['seq']}\n")
        stats = report[name]
        logger.info(
            f"{name:<28} {stats['n_homologs']:>7,} homologs   "
            f"len {stats['length_mean']:.1f} +/- {stats['length_sd']:.1f}   "
            f"qcov {stats['query_coverage_mean']:.2f}   tcov {stats['target_coverage_mean']:.2f}"
        )

    payload = {
        "database": f"OAS ({chain} chains, {n_targets} sequences)",
        "paper_database": "UniRef30 + ColabFold environmental DB (NOT what we searched)",
        "evalue": evalue, "coverage": coverage, "sensitivity": sensitivity,
        "min_identity": min_identity,
        "families": report,
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    write_metrics(metrics_path, payload)
    logger.info(f"wrote {metrics_path} and {len(report)} family FASTAs to {output_folder}")


if __name__ == "__main__":
    typer.run(main)
