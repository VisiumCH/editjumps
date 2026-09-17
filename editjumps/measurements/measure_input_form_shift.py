"""Evaluate joined versus single-domain input-form shift."""
import argparse
import csv
import json
import random
import statistics as st
from pathlib import Path

import numpy as np

from editjumps.core.family_split import disjoint_members, split_family, usable_members
from editjumps.core.generation_metrics import (
    frequencies,
    kl_divergence,
    matrix_agreement,
    mean_levenshtein_to_template,
    mean_pairwise_levenshtein,
    mutual_information_apc,
    one_hot,
    positional_interaction_strength,
    smoothed_composition,
    spectrum_mmd_estimators,
)
from editjumps.core.generation_metrics_undefined import (
    entropy_delta,
    js_divergence,
    js_divergence_positional,
    kl_divergence_positional,
    profile_log_likelihood,
)
from editjumps.core.sequences import PAIR_SEP
from editjumps.pipeline.evaluate.generation_eval import project_to_template

ROOT = Path(__file__).resolve().parents[2]
PAIRS = ROOT / "data/pretrain/oas_homolog_pairs.tsv.gz"
THERA = ROOT / "data/raw/TheraSAbDab_SeqStruc_OnlineDownload.csv"


def read_fasta(path: Path) -> list[str]:
    """Sequences from a FASTA, ignoring ids and leading ``;`` provenance comments."""
    seqs, cur = [], []
    for line in path.read_text().splitlines():
        if line.startswith(";") or not line.strip():
            continue
        if line.startswith(">"):
            if cur:
                seqs.append("".join(cur))
                cur = []
        else:
            cur.append(line.strip())
    if cur:
        seqs.append("".join(cur))
    return seqs


def read_fasta_sets(path: Path) -> dict[str, list[str]]:
    """Group a generated FASTA's records by set name (``model``, each model-free baseline)."""
    sets: dict[str, list[str]] = {}
    name = None
    for line in path.read_text().splitlines():
        if line.startswith(";") or not line.strip():
            continue
        if line.startswith(">"):
            name = line[1:].rsplit("_", 1)[0]
            sets.setdefault(name, [])
        elif name is not None:
            sets[name].append(line.strip())
    return sets


def trastuzumab_light() -> str:
    """Read the real trastuzumab VL from the pinned TheraSAbDab download."""
    with THERA.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["Therapeutic"].strip().lower() == "trastuzumab":
                return row["LightSequence"].strip()
    raise LookupError("trastuzumab absent from the TheraSAbDab download")


def build(family: Path, out_dir: Path, tag: str) -> None:
    """Write the two member-matched arms for one family."""
    vl = trastuzumab_light()
    if PAIR_SEP in vl:
        raise ValueError("the appended light chain must not itself contain the separator")
    members = usable_members(read_fasta(family))
    kept = disjoint_members(members, PAIRS)
    if len(kept) < 220:
        raise ValueError(f"only {len(kept)} members absent from the pairs; need 20 + 200")
    joined = [m + PAIR_SEP + vl for m in kept]
    for j, m in zip(joined, kept, strict=True):
        if j.split(PAIR_SEP)[0] != m:
            raise AssertionError("the arms are not member-matched")
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, seqs in ((f"{tag}_single", kept), (f"{tag}_joined", joined)):
        (out_dir / f"{name}.fasta").write_text(
            "".join(f">{name}_{i}\n{s}\n" for i, s in enumerate(seqs))
        )
    lengths = sorted(len(m) for m in kept)
    print(f"{family.name}: {len(members)} usable, {len(kept)} absent from the pairs "
          f"({100 * len(kept) / len(members):.1f}%); VH {lengths[0]}/{lengths[len(lengths) // 2]}/"
          f"{lengths[-1]}, appended VL {len(vl)} aa")


def rescore(family_fasta: Path, fasta: Path, out: Path, *, truncate: bool, n_templates: int = 20,
            n_variants: int = 20, holdout_size: int = 200, ceiling_n: int = 300, seed: int = 0,
            k: int = 3) -> dict:
    """Score one run's generations, optionally restricted to the VH block."""
    def cut(sequence: str) -> str:
        return sequence.split(PAIR_SEP)[0] if truncate else sequence

    members = disjoint_members(usable_members(read_fasta(family_fasta)), PAIRS)
    split = split_family(members, n_templates, holdout_size, seed)
    holdout = split.reference + split.pool
    rows = [(seq, holdout[i % len(holdout)]) for i, seq in enumerate(split.templates)]
    chosen = random.Random(seed).sample(rows, min(n_templates, len(rows)))

    generated = read_fasta_sets(fasta)["model"]
    if len(generated) != n_templates * n_variants:
        raise ValueError(f"expected {n_templates * n_variants} generations, found {len(generated)}")
    separators = {}
    for seq in generated:
        separators[seq.count(PAIR_SEP)] = separators.get(seq.count(PAIR_SEP), 0) + 1

    reference_template = cut(split.templates[0])
    generated = [cut(s) for s in generated]
    templates = [cut(c[0]) for c in chosen for _ in range(n_variants)]
    natural_all = [cut(s) for s in split.reference]
    partners = [cut(c[1]) for c in chosen]
    pool = [cut(s) for s in split.pool[:ceiling_n]]

    aligned_generated = [project_to_template(s, reference_template) for s in generated]
    aligned_natural = [project_to_template(s, reference_template) for s in partners]
    cap = max(ceiling_n, holdout_size)
    aligned_full = [project_to_template(s, reference_template) for s in natural_all[:cap]]
    aligned_ceiling = [project_to_template(s, reference_template) for s in pool]

    f_i_full, f_ij_full, c_ij_full = frequencies(one_hot(aligned_full))
    strength_full = positional_interaction_strength(c_ij_full)
    mip_full = mutual_information_apc(f_i_full, f_ij_full)
    f_i, f_ij, c_ij = frequencies(one_hot(aligned_generated))
    f_i_ref, f_ij_ref, c_ij_ref = frequencies(one_hot(aligned_natural))
    f_i_c, f_ij_c, c_ij_c = frequencies(one_hot(aligned_ceiling))

    per_template = [{
        "levenshtein_to_template": mean_levenshtein_to_template(
            generated[i * n_variants:(i + 1) * n_variants], templates[i * n_variants]),
        "pairwise_levenshtein": mean_pairwise_levenshtein(
            generated[i * n_variants:(i + 1) * n_variants]),
    } for i in range(len(chosen))]
    mmd = spectrum_mmd_estimators(generated, natural_all, k=k)

    report = {
        "frame": {
            "family_fasta": str(family_fasta),
            "fasta": str(fasta),
            "truncated_at_separator": truncate,
            "alignment_length": len(reference_template),
            "n_generated": len(generated),
            "n_natural_reference": len(natural_all),
            "agreement_reference_n": len(aligned_full),
            "ceiling_n": len(aligned_ceiling),
            "separators_per_generation": separators,
        },
        "defined_by_the_paper": {
            "levenshtein_to_template": float(np.mean([p["levenshtein_to_template"] for p in per_template])),
            # Template-level, because the centre is a mean over TEMPLATES: a sequence-level bootstrap
            # resamples the wrong unit and returns an interval ~2.5x too narrow.
            "levenshtein_to_template_per_template": [
                float(p["levenshtein_to_template"]) for p in per_template],
            "pairwise_levenshtein": float(np.mean([p["pairwise_levenshtein"] for p in per_template])),
            "pairwise_levenshtein_pooled": mean_pairwise_levenshtein(generated),
            "covariance_agreement": matrix_agreement(
                positional_interaction_strength(c_ij), strength_full),
            "mip_agreement": matrix_agreement(mutual_information_apc(f_i, f_ij), mip_full),
            "covariance_agreement_ceiling": matrix_agreement(
                positional_interaction_strength(c_ij_c), strength_full),
            "mip_agreement_ceiling": matrix_agreement(
                mutual_information_apc(f_i_c, f_ij_c), mip_full),
            "kl_generated_vs_natural": kl_divergence(
                smoothed_composition(generated), smoothed_composition(natural_all)),
            "spectrum_mmd": mmd["biased"],
        },
        "our_interpretation": {
            "entropy_delta": entropy_delta(aligned_generated, aligned_natural),
            "js_divergence": js_divergence(generated, natural_all),
            "js_divergence_positional": js_divergence_positional(aligned_generated, aligned_full),
            "kl_divergence_positional": kl_divergence_positional(aligned_generated, aligned_full),
            "profile_log_likelihood": profile_log_likelihood(aligned_generated, aligned_natural),
        },
    }
    # c_ij_ref is computed for parity with the evaluator's own frame; the retired Frobenius keys it
    # feeds are not reported here (see matrix_agreement -- a scalar reduction cannot separate models).
    del f_i_ref, f_ij_ref, c_ij_ref
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    return report


METRICS = (
    ("levenshtein_to_template", "defined_by_the_paper"),
    ("pairwise_levenshtein_pooled", "defined_by_the_paper"),
    ("covariance_agreement", "defined_by_the_paper"),
    ("mip_agreement", "defined_by_the_paper"),
    ("kl_generated_vs_natural", "defined_by_the_paper"),
    ("spectrum_mmd", "defined_by_the_paper"),
    ("entropy_delta", "our_interpretation"),
    ("js_divergence_positional", "our_interpretation"),
    ("kl_divergence_positional", "our_interpretation"),
    ("profile_log_likelihood", "our_interpretation"),
)


def table(metrics_dir: Path) -> None:
    """Print the comparison the section in docs/findings.md quotes."""
    load = lambda name: json.loads((metrics_dir / name).read_text())  # noqa: E731
    arms = {
        "her2 single c40": load("vh-her2vh-single-c40.json"),
        "her2 joined c40": load("vh-her2vh-joined-c40.json"),
        "her2 joined c76": load("vh-her2vh-joined-c76.json"),
        "ty1 single c40": load("vh-ty1-single-c40.json"),
        "ty1 joined c40": load("vh-ty1-joined-c40.json"),
    }
    budget = lambda d: st.mean(d["defined_by_the_paper"]["levenshtein_to_template_per_template"])  # noqa: E731
    print("VH-block edit budget")
    for name, arm in arms.items():
        print(f"  {name:16s} {budget(arm):6.3f}")
    for family, single, joined, extra in (
        ("HER2-VH", "her2 single c40", "her2 joined c40", "her2 joined c76"),
        ("Ty1", "ty1 single c40", "ty1 joined c40", None),
    ):
        print(f"\n{family}: same antibodies, one alignment, only the input form differs")
        header = f"{'metric':30s} {'single':>11s} {'joined':>11s} {'d%':>7s}"
        print(header + (f" {'budget-matched':>14s} {'d%':>7s}" if extra else ""))
        for key, block in METRICS:
            s, j = arms[single][block][key], arms[joined][block][key]
            line = f"{key:30s} {s:11.5f} {j:11.5f} {100 * (j - s) / abs(s):+7.1f}"
            if extra:
                c = arms[extra][block][key]
                line += f" {c:14.5f} {100 * (c - s) / abs(s):+7.1f}"
            print(line)


def main() -> None:
    """Dispatch the three modes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("build", "rescore", "table"), required=True)
    parser.add_argument("--family", type=Path, help="build: the seed family FASTA")
    parser.add_argument("--tag", default="arm", help="build: prefix for the two written FASTAs")
    parser.add_argument("--out-dir", type=Path, help="build: where the arms are written")
    parser.add_argument("--family-fasta", type=Path, help="rescore: the arm's own FASTA")
    parser.add_argument("--fasta", type=Path, help="rescore: the run's generated FASTA")
    parser.add_argument("--truncate", action="store_true", help="rescore: cut at the separator")
    parser.add_argument("--out", type=Path, help="rescore: where the metrics go")
    parser.add_argument("--metrics-dir", type=Path, default=ROOT / "metrics/inputform")
    args = parser.parse_args()
    if args.mode == "build":
        build(args.family, args.out_dir, args.tag)
    elif args.mode == "rescore":
        print(json.dumps(rescore(args.family_fasta, args.fasta, args.out,
                                 truncate=args.truncate), indent=2))
    else:
        table(args.metrics_dir)


if __name__ == "__main__":
    main()
