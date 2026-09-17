"""Measure whether ``homolog_pairs``' per-family cap biases the edit distribution it emits."""

import json
import math
import random
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

from rapidfuzz.distance import Levenshtein

from editjumps.core.cluster_split import cluster_keys
from editjumps.core.utils import get_logger
from editjumps.pipeline.preprocess.pretrain.homolog_pairs import group_families
from editjumps.pipeline.preprocess.pretrain.split_corpus import read_lines

logger = get_logger(__file__)

#: Family-size bins for the "does divergence track family size?" breakdown.
SIZE_BINS: tuple[tuple[int, int], ...] = ((2, 2), (3, 3), (4, 4), (5, 6), (7, 10), (11, 20),
                                          (21, 50), (51, 100), (101, 500), (501, 1 << 30))


def bin_label(size: int) -> str:
    """Name of the `SIZE_BINS` bin holding a family of ``size`` members."""
    for low, high in SIZE_BINS:
        if low <= size <= high:
            return f"{low:04d}-{high}" if high < (1 << 30) else f"{low:04d}+"
    raise ValueError(f"family size {size} fits no bin")


class Distribution:
    """Weighted running summary of one arm's pair edit distances and operation mix."""

    def __init__(self) -> None:
        """Start empty; weights are 1.0 for a capped arm and ``C(k,2)/n_drawn`` for the estimate."""
        self.weight = 0.0
        self.weight_sq = 0.0
        self.total = 0.0
        self.total_sq = 0.0
        self.insert = 0.0
        self.delete = 0.0
        self.substitute = 0.0
        self.histogram: Counter[int] = Counter()

    def add(self, ops: tuple[int, int, int, int], weight: float = 1.0) -> None:
        """Fold in one pair's ``(distance, insertions, deletions, substitutions)``."""
        distance, insert, delete, substitute = ops
        self.weight += weight
        self.weight_sq += weight * weight
        self.total += weight * distance
        self.total_sq += weight * distance * distance
        self.insert += weight * insert
        self.delete += weight * delete
        self.substitute += weight * substitute
        self.histogram[distance] += weight

    def quantile(self, q: float) -> float:
        """Weighted ``q``-quantile of the distance distribution."""
        target, seen = q * self.weight, 0.0
        for distance in sorted(self.histogram):
            seen += self.histogram[distance]
            if seen >= target:
                return float(distance)
        return float("nan")

    def cdf(self) -> dict[int, float]:
        """Weighted empirical CDF, as ``distance -> cumulative fraction``."""
        seen, out = 0.0, {}
        for distance in sorted(self.histogram):
            seen += self.histogram[distance]
            out[distance] = seen / self.weight
        return out

    def summary(self) -> dict[str, float]:
        """Mean, spread, quantiles and operation mix."""
        mean = self.total / self.weight
        variance = max(self.total_sq / self.weight - mean * mean, 0.0)
        ess = self.weight * self.weight / self.weight_sq
        operations = self.insert + self.delete + self.substitute
        return {
            "n_pairs": self.weight,
            "effective_sample_size": ess,
            "mean": mean,
            "sd": math.sqrt(variance),
            "se_mean": math.sqrt(variance / ess),
            "p10": self.quantile(0.10),
            "p25": self.quantile(0.25),
            "median": self.quantile(0.50),
            "p75": self.quantile(0.75),
            "p90": self.quantile(0.90),
            "frac_substitution": self.substitute / operations,
            "frac_indel": (self.insert + self.delete) / operations,
            "frac_insertion": self.insert / operations,
            "frac_deletion": self.delete / operations,
        }


def wasserstein1(left: Distribution, right: Distribution) -> float:
    """Wasserstein-1 distance between two arms, i.e. the area between their CDFs."""
    a, b = left.cdf(), right.cdf()
    keys = sorted(set(a) | set(b))
    area, fa, fb, previous = 0.0, 0.0, 0.0, keys[0]
    for key in keys:
        area += abs(fa - fb) * (key - previous)
        fa, fb, previous = a.get(key, fa), b.get(key, fb), key
    return area


def draw_pairs(size: int, cap: int, rng: random.Random) -> list[tuple[int, int]]:
    """Up to ``cap`` distinct member-index pairs of a family of ``size``, in random order."""
    total = size * (size - 1) // 2
    if total <= cap:
        pairs = list(combinations(range(size), 2))
        rng.shuffle(pairs)
        return pairs
    seen: set[tuple[int, int]] = set()
    pairs = []
    while len(pairs) < cap:
        i, j = sorted(rng.sample(range(size), 2))
        if (i, j) not in seen:
            seen.add((i, j))
            pairs.append((i, j))
    return pairs


def score_pair(left: str, right: str) -> tuple[int, int, int, int]:
    """``(distance, insertions, deletions, substitutions)`` for one pair of sequences."""
    insert = delete = substitute = 0
    for tag, _, _ in Levenshtein.editops(left, right).as_list():
        if tag == "insert":
            insert += 1
        elif tag == "delete":
            delete += 1
        else:
            substitute += 1
    return insert + delete + substitute, insert, delete, substitute


def measure(families: list[list[str]], caps: tuple[int, ...], cap_max: int, seed: int) -> dict:
    """Score every cap in ``caps``, plus an unbiased estimate of the uncapped distribution.

    Args:
        families: Members of each homolog family, one list per family.
        caps: Pairs per family each arm is limited to. None may exceed ``cap_max``.
        cap_max: Pairs actually drawn per family; every arm reads a prefix of these.
        seed: Draw source, so a run is reproducible.

    Returns:
        The per-arm summaries, the weighted uncapped estimate, the distance between them, and a
        breakdown by family size.

    Raises:
        ValueError: If ``caps`` is empty, holds a non-positive cap, or exceeds ``cap_max``.
    """
    # Every arm slices the same `cap_max` draws, so a larger cap runs short and is published under
    # a label it never reached.
    if not caps:
        raise ValueError("caps is empty: there is no arm to compare against the uncapped estimate")
    if min(caps) < 1:
        raise ValueError(f"caps={caps} contains a non-positive cap; every arm needs at least one pair")
    if max(caps) > cap_max:
        raise ValueError(
            f"caps={caps} asks for more pairs per family than cap_max ({cap_max}) ever draws, so the "
            f"cap{max(caps)} arm would measure only {cap_max} pairs and be reported under a label it "
            f"never reached. Raise --cap-max to at least {max(caps)}, or drop that cap."
        )
    rng = random.Random(seed)
    arms = {cap: Distribution() for cap in caps}
    uncapped = Distribution()
    per_bin: dict[str, Distribution] = {}
    per_bin_counts: dict[str, dict[str, int]] = {}
    for done, members in enumerate(families, start=1):
        size = len(members)
        total = size * (size - 1) // 2
        drawn = draw_pairs(size, cap_max, rng)
        scored = [score_pair(members[i], members[j]) for i, j in drawn]
        for cap, arm in arms.items():
            for ops in scored[:cap]:
                arm.add(ops)
        label = bin_label(size)
        bucket = per_bin.setdefault(label, Distribution())
        weight = total / len(scored)
        for ops in scored:
            uncapped.add(ops, weight=weight)
            bucket.add(ops)
        counts = per_bin_counts.setdefault(label, {"families": 0, "pairs_capped": 0, "pairs_uncapped": 0})
        counts["families"] += 1
        counts["pairs_capped"] += min(total, min(caps))
        counts["pairs_uncapped"] += total
        if done % 20000 == 0:
            logger.info(f"  {done}/{len(families)} families scored")

    capped_total = sum(c["pairs_capped"] for c in per_bin_counts.values())
    uncapped_total = sum(c["pairs_uncapped"] for c in per_bin_counts.values())
    breakdown = {}
    for label in sorted(per_bin):
        stats = per_bin[label].summary()
        counts = per_bin_counts[label]
        breakdown[label] = {
            "families": counts["families"],
            "mean_pair_edit_distance": stats["mean"],
            "median": stats["median"],
            "sd": stats["sd"],
            "frac_indel": stats["frac_indel"],
            "share_of_capped_pairs": counts["pairs_capped"] / capped_total,
            "share_of_uncapped_pairs": counts["pairs_uncapped"] / uncapped_total,
        }
    return {
        "arms": {f"cap{cap}": arms[cap].summary() for cap in caps},
        "uncapped_weighted": uncapped.summary(),
        "wasserstein1_to_uncapped": {f"cap{cap}": wasserstein1(arms[cap], uncapped) for cap in caps},
        "totals": {"pairs_at_smallest_cap": capped_total, "pairs_uncapped": uncapped_total},
        "per_family_size": breakdown,
    }


def main() -> None:
    """Cluster the corpus into families, then compare the capped and uncapped pair sets."""
    import typer

    def run(
        output_folder: Path = Path("data/pretrain"),
        corpus_name: str = "oas_corpus.txt.gz",
        cdr_keys_name: str = "oas_corpus.cdr_keys.tsv.gz",
        min_seq_id: float = 0.5,
        coverage: float = 0.8,
        mmseqs_mode: str = "easy-cluster",
        split_memory_limit: str = "8G",
        caps: str = "20,50,200,1000",
        cap_max: int = 1000,
        max_lines: int = 0,
        seed: int = 0,
        metrics_path: Path = Path("metrics/homolog_cap_bias.json"),
    ) -> None:
        """Write the cap-vs-uncapped comparison to ``metrics_path`` and log its headline."""
        lines = read_lines(output_folder / corpus_name)
        rows = read_lines(output_folder / cdr_keys_name)
        if len(lines) != len(rows):
            raise ValueError(f"corpus/keys length mismatch: {len(lines)} vs {len(rows)}")
        if max_lines:
            lines, rows = lines[:max_lines], rows[:max_lines]
        cluster_ids = cluster_keys(
            [row.replace("\t", "") for row in rows], min_seq_id=min_seq_id, coverage=coverage,
            mode=mmseqs_mode, tmp_dir=output_folder / "mmseqs_tmp_cap_bias",
            split_memory_limit=split_memory_limit, stage="build_homolog_pairs",
        )
        families = [m for m in group_families(cluster_ids, lines).values() if len(m) >= 2]
        logger.info(f"{len(lines)} lines -> {len(families)} families with >=2 members")
        cap_values = tuple(sorted(int(c) for c in caps.split(",") if c.strip()))
        report = measure(families, cap_values, cap_max, seed)
        report["build"] = {"corpus": corpus_name, "lines": len(lines), "families": len(families),
                           "min_seq_id": min_seq_id, "coverage": coverage, "mmseqs_mode": mmseqs_mode,
                           "cap_max": cap_max, "seed": seed}
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(report, indent=1))
        smallest = report["arms"][f"cap{cap_values[0]}"]
        logger.info(f"cap {cap_values[0]}: mean edit distance {smallest['mean']:.2f} over "
                    f"{smallest['n_pairs']:.0f} pairs; uncapped estimate "
                    f"{report['uncapped_weighted']['mean']:.2f} +/- "
                    f"{report['uncapped_weighted']['se_mean']:.2f} over "
                    f"{report['totals']['pairs_uncapped']} -> wrote {metrics_path}")

    typer.run(run)


if __name__ == "__main__":
    sys.exit(main())
