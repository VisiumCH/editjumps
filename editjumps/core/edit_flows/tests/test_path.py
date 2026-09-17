"""The aligned pair the process is defined on, and the mixture drawn between its ends."""


def test_sample_edit_path_recovers_target_with_requested_edits() -> None:
    """Sample_edit_path yields z_1==target and z_0==noise, related by the requested edits."""
    import random

    from editjumps.core.edit_flows.path import EPS, sample_edit_path, strip_epsilon

    rng = random.Random(0)
    target = [5, 6, 7, 8]
    z0, z1 = sample_edit_path(target, vocab_size=20, target_n_sub=2, target_n_del=1, rng=rng)
    assert len(z0) == len(z1)
    assert strip_epsilon(z1) == target  # target reconstructed exactly, in order
    # columns: 2 substitutions + 1 deletion + (len-2)=2 insertions
    n_ins = sum(1 for a in z0 if a == EPS)
    n_del = sum(1 for b in z1 if b == EPS)
    n_sub = sum(1 for a, b in zip(z0, z1, strict=True) if a != EPS and b != EPS)
    assert (n_ins, n_del, n_sub) == (2, 1, 2)
    assert len(strip_epsilon(z0)) == n_del + n_sub  # source has one token per del/sub

def test_mixture_path_endpoints_are_source_and_target() -> None:
    """Kappa=0 returns z_0 exactly; kappa=1 returns z_1 exactly."""
    import random

    from editjumps.core.edit_flows.path import EPS, mixture_path

    z0 = [1, EPS, 3]
    z1 = [1, 2, EPS]
    assert mixture_path(z0, z1, kappa=0.0, rng=random.Random(1)) == z0
    assert mixture_path(z0, z1, kappa=1.0, rng=random.Random(1)) == z1
