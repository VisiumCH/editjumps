"""The aligned pair ``(z_0, z_1)`` and its time-``t`` interpolant (Edit Flows arXiv 2506.09018, eq.."""

import random

#: The gap sentinel ("epsilon" in the papers), and DEVIATION 1's whole mechanism.
EPS = -1


def sample_edit_path(
    target: list[int], vocab_size: int, target_n_sub: int, target_n_del: int, rng: random.Random
) -> tuple[list[int], list[int]]:
    """Build a noise->data aligned pair (Meta's reference ``get_z``), for toy and unconditional runs."""
    length = len(target)
    n_sub = min(length, target_n_sub)
    n_del = target_n_del + target_n_sub - n_sub  # shortfall -> extra deletions
    n_ins = length - n_sub

    source_tokens = [rng.randrange(vocab_size) for _ in range(n_del + n_sub)]
    # Column markers: "ins" = source gap (EPS in z_0), "del" = target gap (EPS in
    # z_1), "sub" = both real. Shuffled to interleave the edits.
    markers = ["ins"] * n_ins + ["del"] * n_del + ["sub"] * n_sub
    rng.shuffle(markers)

    z_0: list[int] = []
    z_1: list[int] = []
    ti = 0  # index into target
    si = 0  # index into source_tokens
    for marker in markers:
        if marker == "del":
            z_0.append(source_tokens[si])
            z_1.append(EPS)
            si += 1
        elif marker == "ins":
            z_0.append(EPS)
            z_1.append(target[ti])
            ti += 1
        else:  # "sub"
            z_0.append(source_tokens[si])
            z_1.append(target[ti])
            si += 1
            ti += 1
    return z_0, z_1


def mixture_path(z_0: list[int], z_1: list[int], kappa: float, rng: random.Random) -> list[int]:
    """Sample the time-``t`` interpolant ``z_t``: each column takes ``z_1`` with probability ``kappa``."""
    return [t1 if rng.random() < kappa else t0 for t0, t1 in zip(z_0, z_1, strict=True)]


def strip_epsilon(z: list[int]) -> list[int]:
    """Drop gap sentinels to recover a real token sequence."""
    return [token for token in z if token != EPS]
