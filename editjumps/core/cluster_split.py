"""Cluster-based, leakage-aware train/validation split via MMseqs2."""

import random
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

from editjumps.core.capacity import require_local_capacity
from editjumps.core.utils import get_logger

logger = get_logger(__file__)


def cluster_keys(
    keys: Sequence[str],
    *,
    min_seq_id: float = 0.9,
    coverage: float = 0.8,
    cov_mode: int = 1,
    mode: str = "easy-cluster",
    tmp_dir: Path = Path("/tmp/mmseqs"),
    mmseqs_bin: str = "mmseqs",
    split_memory_limit: str = "8G",
    stage: str = "split_corpus",
) -> list[int]:
    """Cluster sequences with MMseqs2 and return a cluster id per input."""
    if shutil.which(mmseqs_bin) is None:
        raise FileNotFoundError(
            f"MMseqs2 binary {mmseqs_bin!r} not found on PATH. "
            "Install with `brew install mmseqs2` or `conda install -c bioconda mmseqs2`."
        )

    n = len(keys)
    require_local_capacity(n, f"MMseqs {mode} clustering", stage=stage)
    # dedup identical keys: cluster the unique set, map members back afterwards
    uniq: dict[str, int] = {}
    for k in keys:
        uniq.setdefault(k, len(uniq))
    logger.info(f"{n} keys -> {len(uniq)} unique; clustering with mmseqs {mode} (min_seq_id={min_seq_id})")

    tmp_dir.mkdir(parents=True, exist_ok=True)
    fasta = tmp_dir / "keys.fasta"
    with open(fasta, "w") as fh:
        for key, uid in uniq.items():
            fh.write(f">{uid}\n{key}\n")

    result_prefix = tmp_dir / "clu"
    cmd = [
        mmseqs_bin,
        mode,
        str(fasta),
        str(result_prefix),
        str(tmp_dir / "work"),
        "--min-seq-id",
        str(min_seq_id),
        "-c",
        str(coverage),
        "--cov-mode",
        str(cov_mode),
        "--kmer-per-seq",
        "100",
        # Bound the prefilter working set so a full-corpus run cannot swap the machine to death.
        "--split-memory-limit",
        split_memory_limit,
    ]
    logger.info(f"running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, capture_output=True, text=True)

    # easy-cluster/linclust write <prefix>_cluster.tsv: representative<TAB>member (uid strings)
    rep_of_uid: dict[int, int] = {}
    with open(f"{result_prefix}_cluster.tsv") as fh:
        for line in fh:
            rep, member = line.rstrip("\n").split("\t")
            rep_of_uid[int(member)] = int(rep)

    # remap representatives to contiguous cluster ids; unseen uids -> own singleton
    remap: dict[int, int] = {}
    next_id = len(remap)

    def cluster_for_uid(uid: int) -> int:
        nonlocal next_id
        rep = rep_of_uid.get(uid, uid)  # dropped keys cluster alone
        if rep not in remap:
            remap[rep] = next_id
            next_id += 1
        return remap[rep]

    uid_cluster = {uid: cluster_for_uid(uid) for uid in uniq.values()}
    out = [uid_cluster[uniq[k]] for k in keys]
    logger.info(f"{n} keys -> {len(set(out))} clusters")
    return out


def split_by_cluster(cluster_ids: Sequence[int], val_frac: float = 0.05) -> list[str]:
    """Assign whole clusters to ``train``/``val`` to approach ``val_frac`` by item count."""
    groups: dict[int, list[int]] = {}
    for idx, c in enumerate(cluster_ids):
        groups.setdefault(c, []).append(idx)
    order = sorted(groups, key=lambda c: (-len(groups[c]), c))

    n = len(cluster_ids)
    val: set[int] = set()
    n_val = 0
    for comp in order:
        idxs = groups[comp]
        # place in val only if it moves the achieved fraction toward the target
        if n and abs((n_val + len(idxs)) / n - val_frac) < abs(n_val / n - val_frac):
            val.update(idxs)
            n_val += len(idxs)

    out = ["train"] * n
    for i in val:
        out[i] = "val"
    logger.info(f"split: {n - n_val} train / {n_val} val ({n_val / n:.3f} val frac)" if n else "split: empty")
    return out


def split_pairs_by_family(
    pairs: Sequence[tuple[str, ...]], val_frac: float, seed: int = 0
) -> tuple[list[int], list[int]]:
    """Split pair indices into train/val so no sequence appears on both sides."""
    if val_frac <= 0:
        return list(range(len(pairs))), []

    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for x0, x1 in ((p[0], p[1]) for p in pairs):
        union(x0, x1)

    # Assign whole components to val until the target fraction is reached.
    by_component: dict[str, list[int]] = {}
    for i, pair in enumerate(pairs):
        by_component.setdefault(find(pair[0]), []).append(i)
    components = sorted(by_component)  # sorted first so the shuffle is reproducible
    random.Random(seed).shuffle(components)

    target = int(len(pairs) * val_frac)
    val_idx: list[int] = []
    for component in components:
        if len(val_idx) >= target:
            break
        val_idx.extend(by_component[component])

    val_set = set(val_idx)
    train_idx = [i for i in range(len(pairs)) if i not in val_set]

    # Enforce the guarantee rather than trusting the construction: any val pair sharing a sequence with train.
    train_seqs = {s for i in train_idx for s in (pairs[i][0], pairs[i][1])}
    leaked = [i for i in val_idx if pairs[i][0] in train_seqs or pairs[i][1] in train_seqs]
    if leaked:
        leaked_set = set(leaked)
        val_idx = [i for i in val_idx if i not in leaked_set]
        train_idx = sorted(train_idx + leaked)
    return train_idx, val_idx
