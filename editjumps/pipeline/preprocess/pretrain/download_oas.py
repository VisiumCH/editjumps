"""Download OAS antibody sequences and build a pretraining corpus (unlabeled)."""

import gzip
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated

import pandas as pd
import requests
import typer

from editjumps.core.utils import get_logger

logger = get_logger(__file__)

AA = set("ACDEFGHIKLMNPQRSTVWY")
OAS_NGSDB = "https://opig.stats.ox.ac.uk/webapps/ngsdb"
# AA variable-region columns, in detection order: unpaired first, then paired chains
SEQ_COLS = ["sequence_alignment_aa", "sequence_alignment_aa_heavy", "sequence_alignment_aa_light"]
# paired CDR-3 columns, for the leakage-aware split keys
CDR3_COLS = ("cdr3_aa_heavy", "cdr3_aa_light")
OUTPUT_NAME = "oas_corpus.txt.gz"
CDR_KEYS_NAME = "oas_corpus.cdr_keys.tsv.gz"


def list_index(url: str) -> list[str]:
    """Return the hrefs listed in an OAS open directory index page."""
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    hrefs = re.findall(r'href="([^"?]+)"', r.text)
    return [h for h in hrefs if not h.startswith((".", "/", "http"))]


def crawl_units(collection: str, studies: list[str] | None = None) -> list[str]:
    """Crawl the OAS directory index to list data-unit URLs."""
    base = f"{OAS_NGSDB}/{collection}/"
    study_dirs = [h.strip("/") for h in list_index(base) if h.endswith("/")]
    if studies:
        wanted = {s.lower() for s in studies}
        study_dirs = [s for s in study_dirs if s.lower() in wanted]
    logger.info(f"crawling {len(study_dirs)} {collection} study folder(s)")
    urls: list[str] = []
    for study in study_dirs:
        study_url = f"{base}{study}/"
        try:
            # the CSV subfolder is 'csv/' (older studies) or 'csv_paired/' (newer); discover it
            csv_dirs = [h for h in list_index(study_url) if h.endswith("/") and "csv" in h.lower()]
            for sub in csv_dirs:
                csv_idx = f"{study_url}{sub}"
                urls += [csv_idx + f for f in list_index(csv_idx) if f.endswith(".csv.gz")]
        except requests.RequestException as exc:
            logger.warning(f"skipping {study}: {exc}")
    return urls


def parse_bulk_script(bulk_script: Path) -> list[str]:
    """Extract the data-unit URLs from an OAS ``bulk_download.sh``."""
    return re.findall(r"https?://\S+\.csv\.gz", bulk_script.read_text())


def download_unit(url: str, dest_dir: Path) -> Path | None:
    """Download one OAS data unit, skipping if already present."""
    dest = dest_dir / url.rsplit("/", 1)[-1]
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    try:
        with requests.get(url, stream=True, timeout=300) as r:
            r.raise_for_status()
            with open(dest, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
    except requests.RequestException as exc:
        logger.warning(f"failed {url}: {exc}")
        dest.unlink(missing_ok=True)
        return None
    return dest


def clean_seq(raw: object, min_len: int, max_len: int) -> str | None:
    """Strip IMGT gaps and validate one amino-acid sequence."""
    seq = re.sub(r"[.\-\s]", "", str(raw)).upper()
    if min_len <= len(seq) <= max_len and set(seq) <= AA:
        return seq
    return None


def clean_cdr(raw: object) -> str:
    """Strip IMGT gaps/whitespace from a CDR cell and uppercase it (no length bounds)."""
    if not isinstance(raw, str):  # missing cells read as float NaN -> "NAN" would pass the AA check
        return ""
    cdr = re.sub(r"[.\-\s]", "", raw).upper()
    return cdr if cdr and set(cdr) <= AA else ""


def iter_sequences(unit_path: Path, min_len: int = 70, max_len: int = 200) -> Iterator[str]:
    """Yield cleaned amino-acid variable-region sequences (one per chain)."""
    # header=1 -> row 0 is the JSON metadata line, row 1 is the real CSV header
    df = pd.read_csv(unit_path, compression="gzip", header=1, low_memory=False)
    cols = [c for c in SEQ_COLS if c in df.columns]
    if not cols:
        logger.warning(f"{unit_path.name}: no known sequence column in {list(df.columns)[:8]}...")
        return
    for col in cols:
        for raw in df[col].dropna():
            seq = clean_seq(raw, min_len, max_len)
            if seq is not None:
                yield seq


def iter_pairs(unit_path: Path, sep: str, min_len: int = 70, max_len: int = 200) -> Iterator[tuple[str, str, str]]:
    """Yield ``(VH<sep>VL, CDR-H3, CDR-L3)`` from a *paired* data unit (one per antibody)."""
    df = pd.read_csv(unit_path, compression="gzip", header=1, low_memory=False)
    hc, lc = "sequence_alignment_aa_heavy", "sequence_alignment_aa_light"
    if hc not in df.columns or lc not in df.columns:
        logger.warning(f"{unit_path.name}: not paired (missing {hc}/{lc}); skipping for --pair-chains")
        return
    h3c, l3c = CDR3_COLS
    have_cdr = h3c in df.columns and l3c in df.columns
    h3col = df[h3c] if have_cdr else [None] * len(df)
    l3col = df[l3c] if have_cdr else [None] * len(df)
    for raw_h, raw_l, raw_h3, raw_l3 in zip(df[hc], df[lc], h3col, l3col, strict=False):
        vh, vl = clean_seq(raw_h, min_len, max_len), clean_seq(raw_l, min_len, max_len)
        if vh is not None and vl is not None:
            yield f"{vh}{sep}{vl}", clean_cdr(raw_h3), clean_cdr(raw_l3)


def build_corpus(
    urls: list[str],
    output_folder: Path,
    max_units: int | None = None,
    min_len: int = 70,
    max_len: int = 200,
    pair_chains: bool = False,
    pair_sep: str = ".",
    emit_cdr_keys: bool = False,
) -> Path:
    """Download OAS units and write a deduplicated MLM pretraining corpus."""
    if emit_cdr_keys and not pair_chains:
        raise ValueError("emit_cdr_keys requires pair_chains (CDR-L3 needs the light chain)")
    output_folder.mkdir(parents=True, exist_ok=True)
    units_dir = output_folder / "oas_units"
    units_dir.mkdir(exist_ok=True)
    if max_units is not None:
        urls = urls[:max_units]
    logger.info(f"{len(urls)} data unit(s) to process")

    corpus_path = output_folder / OUTPUT_NAME
    keys_path = output_folder / CDR_KEYS_NAME
    seen: set[str] = set()  # in-memory dedup; dedup externally for very large runs
    kept = 0
    with gzip.open(corpus_path, "wt") as out:
        keys_out = gzip.open(keys_path, "wt") if emit_cdr_keys else None
        try:
            for i, url in enumerate(urls, 1):
                unit = download_unit(url, units_dir)
                if unit is None:
                    continue
                if pair_chains:
                    rows: Iterator[tuple[str, str, str]] = iter_pairs(unit, pair_sep, min_len, max_len)
                else:
                    rows = ((seq, "", "") for seq in iter_sequences(unit, min_len, max_len))
                for seq, cdrh3, cdrl3 in rows:
                    if seq not in seen:
                        seen.add(seq)
                        out.write(seq + "\n")
                        if keys_out is not None:
                            keys_out.write(f"{cdrh3}\t{cdrl3}\n")
                        kept += 1
                logger.info(f"[{i}/{len(urls)}] {unit.name}: corpus now {kept} unique sequences")
        finally:
            if keys_out is not None:
                keys_out.close()

    logger.info(f"Wrote {corpus_path}: {kept} unique antibody sequences for MLM pretraining")
    if emit_cdr_keys:
        logger.info(f"Wrote {keys_path}: CDR-H3/L3 keys aligned to corpus lines (for split_corpus)")
    return corpus_path


def main(
    output_folder: Annotated[Path, typer.Option()] = Path("data/pretrain"),
    collection: Annotated[str, typer.Option(help="OAS collection to crawl: paired or unpaired")] = "paired",
    study: Annotated[list[str] | None, typer.Option(help="Study name(s) to include; omit for all")] = None,
    bulk_script: Annotated[Path | None, typer.Option(help="Use an OAS bulk_download.sh instead of crawling")] = None,
    max_units: Annotated[int | None, typer.Option(help="Cap data units (testing); omit for all")] = None,
    min_len: Annotated[int, typer.Option()] = 70,
    max_len: Annotated[int, typer.Option()] = 200,
    pair_chains: Annotated[bool, typer.Option(help="Emit VH<sep>VL per antibody (paired units)")] = False,
    pair_sep: Annotated[
        str, typer.Option(help="Separator between chains for --pair-chains")
    ] = ".",
    emit_cdr_keys: Annotated[
        bool, typer.Option(help="Also write CDR-H3/L3 keys per line for split_corpus (needs --pair-chains)")
    ] = False,
) -> None:
    """CLI entry point for building the OAS pretraining corpus."""
    urls = parse_bulk_script(bulk_script) if bulk_script else crawl_units(collection, study)
    build_corpus(urls, output_folder, max_units, min_len, max_len, pair_chains, pair_sep, emit_cdr_keys)


if __name__ == "__main__":
    typer.run(main)
