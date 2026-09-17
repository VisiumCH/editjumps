"""EvoFlows' evaluation metrics (Appendix B), implemented from their definitions."""

import math

import numpy as np

# 20 standard amino acids plus the gap, in a fixed order. Index 20 is the gap, matching the paper's
# "20 standard amino acids plus gap" (B.4).
ALPHABET = "ACDEFGHIKLMNPQRSTVWY-"
GAP_INDEX = 20
N_SYMBOLS = len(ALPHABET)

# BLOSUM62 background amino-acid frequencies (Altschul et al.), the paper's smoothing prior in place of a.
BLOSUM62_BACKGROUND = {
    "A": 0.074, "C": 0.025, "D": 0.054, "E": 0.054, "F": 0.047,
    "G": 0.074, "H": 0.026, "I": 0.068, "K": 0.058, "L": 0.099,
    "M": 0.025, "N": 0.045, "P": 0.039, "Q": 0.034, "R": 0.052,
    "S": 0.057, "T": 0.051, "V": 0.073, "W": 0.013, "Y": 0.032,
}
# "We calibrated the gap frequency mu_GAP empirically ... finding a value of 0.7%" (B.4).
GAP_BACKGROUND = 0.007


def background_frequencies() -> np.ndarray:
    """Return the smoothing prior over `ALPHABET`, normalised to sum to 1."""
    prior = np.array([BLOSUM62_BACKGROUND.get(a, 0.0) for a in ALPHABET[:GAP_INDEX]] + [GAP_BACKGROUND])
    return prior / prior.sum()


def one_hot(aligned: list[str]) -> np.ndarray:
    """One-hot encode equal-length aligned sequences as the paper's ``X`` (eq 17)."""
    if not aligned:
        return np.zeros((0, 0, N_SYMBOLS))
    lengths = {len(s) for s in aligned}
    if len(lengths) != 1:
        raise ValueError(f"sequences must be aligned to one length; got {sorted(lengths)[:5]}")
    index = {a: i for i, a in enumerate(ALPHABET)}
    encoded = np.zeros((len(aligned), len(aligned[0]), N_SYMBOLS))
    for n, seq in enumerate(aligned):
        for i, char in enumerate(seq):
            encoded[n, i, index.get(char, GAP_INDEX)] = 1.0
    return encoded


def frequencies(encoded: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the paper's single, joint and covariance tensors (eq 17). ``f_i(a) = 1/N sum_n x_nia``."""
    n = max(len(encoded), 1)
    f_i = encoded.sum(axis=0) / n                                   # (L, 21)
    f_ij = np.einsum("nia,njb->ijab", encoded, encoded) / n         # (L, L, 21, 21)
    c_ij = f_ij - np.einsum("ia,jb->ijab", f_i, f_i)
    return f_i, f_ij, c_ij


def positional_interaction_strength(c_ij: np.ndarray) -> np.ndarray:
    """Contract the 4D covariance to a per-position-pair coupling (eq 18)."""
    return np.sqrt((c_ij ** 2).sum(axis=(2, 3)))


def mutual_information_apc(f_i: np.ndarray, f_ij: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Mutual information with average product correction, MIp (eq 19–22). ``I_ij = sum_ab f_ij log(f_ij."""
    outer = np.einsum("ia,jb->ijab", f_i, f_i)
    ratio = np.where(f_ij > 0, f_ij / np.maximum(outer, eps), 1.0)
    information = (f_ij * np.log(np.maximum(ratio, eps))).sum(axis=(2, 3))   # (L, L)
    row = information.mean(axis=1)
    col = information.mean(axis=0)
    total = information.mean()
    apc = np.outer(row, col) / max(total, eps)
    return information - apc


def matrix_agreement(generated: np.ndarray, natural: np.ndarray, min_separation: int = 5) -> float:
    """Pearson correlation between two ``(L, L)`` coupling matrices, off the diagonal band."""
    if generated.shape != natural.shape:
        raise ValueError(f"shapes differ: {generated.shape} vs {natural.shape}")
    length = generated.shape[0]
    rows, cols = np.triu_indices(length, k=min_separation)
    if rows.size < 2:
        return float("nan")
    a, b = generated[rows, cols], natural[rows, cols]
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def smoothed_composition(sequences: list[str], alpha: float = 1.0) -> np.ndarray:
    """Amino-acid composition with the paper's BLOSUM62 additive smoothing (eq 24)."""
    index = {a: i for i, a in enumerate(ALPHABET)}
    counts = np.zeros(N_SYMBOLS)
    for seq in sequences:
        for char in seq:
            counts[index.get(char, GAP_INDEX)] += 1
    prior = background_frequencies()
    return (counts + alpha * prior) / (counts.sum() + alpha)


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """``D_KL(p || q)`` over amino-acid frequencies (eq 23), asymmetric as the paper notes. **Eq 23."""
    p = p / p.sum()
    q = q / q.sum()
    return float((p * np.log(np.maximum(p, eps) / np.maximum(q, eps))).sum())


def spectrum_features(sequence: str, k: int = 3) -> dict[str, int]:
    """Count contiguous k-mers, the spectrum-kernel feature map ``Phi_k`` (eq 26)."""
    counts: dict[str, int] = {}
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i + k]
        counts[kmer] = counts.get(kmer, 0) + 1
    return counts


def spectrum_kernel(a: dict[str, int], b: dict[str, int]) -> float:
    """Inner product of two k-mer count vectors (eq 27)."""
    small, large = (a, b) if len(a) <= len(b) else (b, a)
    return float(sum(count * large.get(kmer, 0) for kmer, count in small.items()))


def spectrum_mmd_estimators(generated: list[str], reference: list[str], k: int = 3) -> dict:
    """Both MMD estimators under the spectrum kernel (eq 25), from one pass over the kernels."""
    if not generated or not reference:
        return {"biased": 0.0, "unbiased_squared": 0.0, "n_generated": 0, "n_reference": 0}
    gen = [spectrum_features(s, k) for s in generated]
    ref = [spectrum_features(s, k) for s in reference]

    def within(xs: list[dict]) -> tuple[float, float]:
        """Total kernel mass including and excluding the diagonal."""
        total = diagonal = 0.0
        for i, x in enumerate(xs):
            for j, y in enumerate(xs):
                value = spectrum_kernel(x, y)
                total += value
                if i == j:
                    diagonal += value
        return total, diagonal

    m, n = len(gen), len(ref)
    gen_total, gen_diagonal = within(gen)
    ref_total, ref_diagonal = within(ref)
    cross = sum(spectrum_kernel(x, y) for x in gen for y in ref)

    biased_squared = gen_total / (m * m) + ref_total / (n * n) - 2 * cross / (m * n)
    # The U-statistic needs at least two samples per side to have an off-diagonal at all.
    if m > 1 and n > 1:
        unbiased_squared = (
            (gen_total - gen_diagonal) / (m * (m - 1))
            + (ref_total - ref_diagonal) / (n * (n - 1))
            - 2 * cross / (m * n)
        )
    else:
        unbiased_squared = float("nan")

    return {
        "biased": math.sqrt(max(biased_squared, 0.0)),
        "unbiased_squared": unbiased_squared,
        "n_generated": m,
        "n_reference": n,
    }


def spectrum_mmd(generated: list[str], reference: list[str], k: int = 3) -> float:
    """Return the paper's biased-V-statistic MMD (eq 25)."""
    return spectrum_mmd_estimators(generated, reference, k)["biased"]


def levenshtein(a: str, b: str) -> int:
    """Edit distance between two sequences (iterative two-row DP)."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def mean_levenshtein_to_template(generated: list[str], template: str) -> float:
    """Average edit distance from each generated sequence to the starting sequence x0."""
    if not generated:
        return 0.0
    return float(np.mean([levenshtein(s, template) for s in generated]))


def mean_pairwise_levenshtein(generated: list[str], max_pairs: int = 2000, seed: int = 0) -> float:
    """Average edit distance between distinct generated sequences — their diversity axis."""
    if len(generated) < 2:
        return 0.0
    rng = np.random.default_rng(seed)
    pairs = [(i, j) for i in range(len(generated)) for j in range(i + 1, len(generated))]
    if len(pairs) > max_pairs:
        pairs = [pairs[i] for i in rng.choice(len(pairs), size=max_pairs, replace=False)]
    return float(np.mean([levenshtein(generated[i], generated[j]) for i, j in pairs]))
