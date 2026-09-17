"""The MSA proposer's wire protocol to the other environment, driven against a fake runner."""

from pathlib import Path

import pytest

from editjumps.test.fakes import FAKE_EVODIFF_RUNNER, _fake_shim


def test_evodiff_runner_failure_surfaces_its_stderr_not_a_blank_pipe(tmp_path: Path) -> None:
    """A failure in the OTHER environment must arrive as a message, not as an empty response."""
    from editjumps.core.evodiff_msa.alignment import write_a3m
    from editjumps.core.evodiff_msa.proposer import MsaProposer

    a3m = tmp_path / "family.a3m"
    write_a3m(a3m, "ACDEFGHIKL", ["ACDEFGHIKM"] * 4)

    dies = 'import sys\nprint("boom: no module named evodiff", file=sys.stderr)\nsys.exit(1)\n'
    shim, _log = _fake_shim(tmp_path / "dead", dies)
    with pytest.raises(RuntimeError, match="boom: no module named evodiff"):
        with MsaProposer(a3m, 0, n_sequences=4, shim=shim):
            pass

    # An error response is raised, not returned: too many rows must not become a smaller MSA.
    shim, _log = _fake_shim(tmp_path / "ok")
    with pytest.raises(RuntimeError, match="msa num_seqs"):
        with MsaProposer(a3m, 0, n_sequences=64, shim=shim):
            pass

    with pytest.raises(ValueError, match="must be used as a context manager"):
        MsaProposer(a3m, 0)(["A"], 0, "")
    with pytest.raises(ValueError, match="model="):
        MsaProposer(a3m, 0, model="msa-d3pm-blosum")
    with pytest.raises(ValueError, match="an MSA of one row"):
        MsaProposer(a3m, 0, n_sequences=1)


def test_evodiff_refuses_an_alignment_that_was_replaced_under_it(tmp_path: Path) -> None:
    """A clobbered a3m is a loud failure, not a silent generation from the wrong family."""
    from editjumps.core.evodiff_msa.alignment import write_a3m
    from editjumps.core.evodiff_msa.proposer import MsaProposer

    a3m = tmp_path / "template_0000.a3m"
    write_a3m(a3m, "ACDEFGHIKL", ["ACDEFGHIKM"] * 4)
    shim, _log = _fake_shim(tmp_path / "ok")
    # Same width, different sequence: the other family's alignment, landed at the same path.
    write_a3m(a3m, "MMMMMMMMMM", ["ACDEFGHIKM"] * 4)
    with pytest.raises(ValueError, match="a different query than the one written"):
        with MsaProposer(a3m, 0, n_sequences=4, shim=shim, expect_query="ACDEFGHIKL"):
            pass
    # And the honest case still loads.
    with MsaProposer(a3m, 0, n_sequences=4, shim=shim, expect_query="MMMMMMMMMM") as propose:
        assert propose.length == 10


def test_evodiff_protocol_is_implemented_by_both_the_runner_and_the_fake() -> None:
    """The wire protocol is documented once; three implementations must agree on it."""
    import re

    from editjumps.core.evodiff_msa.proposer import MODELS, PROTOCOL

    runner = Path("editjumps/core/evodiff_msa/evodiff_msa_runner.py").read_text()
    assert set(re.findall(r'op == "(\w+)"', runner)) == set(PROTOCOL)
    assert set(re.findall(r'op == "(\w+)"', FAKE_EVODIFF_RUNNER)) == set(PROTOCOL)
    # Only order-agnostic checkpoints: D3PM MSA models decode a whole row over a fixed timestep
    # schedule, so they have no partial-row form and cannot be budget-matched.
    assert set(re.findall(r'"(msa-[a-z0-9-]+)":', runner)) == set(MODELS)
    # The runner must not import this repo: it runs under an interpreter that does not have it.
    assert not re.search(r"^\s*(from|import)\s+editjumps\b", runner, re.M)
