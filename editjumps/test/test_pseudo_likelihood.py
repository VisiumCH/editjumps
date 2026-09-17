"""The pseudo-log-likelihood estimator: per-sequence values, not only their mean."""

import pytest


def test_pseudo_log_likelihood_returns_per_sequence_values() -> None:
    """The estimator must expose the per-sequence numbers, not only their mean."""
    from editjumps.core.pseudo_likelihood import pseudo_log_likelihood

    empty = pseudo_log_likelihood([])
    assert set(empty) == {"mean", "sum_mean", "n", "per_sequence", "per_sequence_sum"}
    assert empty["per_sequence"] == [] and empty["n"] == 0


def test_scoring_without_torch_explains_the_missing_group_rather_than_tracebacking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A lean install must not answer `rank` with a bare ImportError. `edit` guards its equivalent."""
    import sys

    from editjumps.core import pseudo_likelihood

    # None in sys.modules is the documented way to make an import fail without touching builtins.
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ImportError) as caught:
        pseudo_likelihood.pseudo_log_likelihood(["ACDEF"], "facebook/esm2_t12_35M_UR50D")

    message = str(caught.value)
    assert "--group train" in message, "the message must name the fix"
    assert "uv sync" in message
    assert "2.5 GB" in message or "--model" in message, "warn about the download it triggers"
