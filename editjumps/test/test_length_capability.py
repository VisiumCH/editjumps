"""Which §4.2 methods can enter a growth benchmark at all, and what each declares."""

import random
from pathlib import Path

import pytest

from editjumps.core.edit_flows.inference import RateField
from editjumps.test.fakes import _stub_proposer


def test_every_evaluator_declares_what_it_can_do_to_a_sequences_length() -> None:
    """All three evaluators declare a length capability, and the declarations are internally sound."""
    from editjumps.core.length_capability import (
        NATIVE,
        REFUSED,
        SCAFFOLDED,
        LengthCapability,
        LengthCapabilityError,
    )
    from editjumps.pipeline.evaluate import (
        evodiff_msa_baseline,
        evotune_baseline,
        generation_eval,
    )

    for module in (generation_eval, evotune_baseline, evodiff_msa_baseline):
        name = module.__name__.rsplit(".", 1)[-1]
        assert isinstance(getattr(module, "CAN_CHANGE_LENGTH", None), bool), (
            f"{name} declares no CAN_CHANGE_LENGTH; a growth benchmark cannot tell whether this "
            "row could have played"
        )
        capability = getattr(module, "LENGTH_CAPABILITY", None)
        assert isinstance(capability, LengthCapability), f"{name} declares no LENGTH_CAPABILITY"
        assert capability.can_change_length is module.CAN_CHANGE_LENGTH, (
            f"{name}: the flag and the record disagree, so a reader of either is misinformed"
        )
        assert capability.target_length in (NATIVE, SCAFFOLDED, REFUSED)
        assert len(capability.mechanism) > 40, f"{name}: the mechanism must name what it is"
        # A capability with no target requested must never refuse: that is what keeps every existing
        # invocation working unchanged.
        capability.check(None)

    # The audited split: the editor is the ONLY method whose own kernel changes length.
    assert generation_eval.CAN_CHANGE_LENGTH is True
    assert generation_eval.LENGTH_CAPABILITY.target_length == NATIVE
    assert evotune_baseline.CAN_CHANGE_LENGTH is False
    assert evodiff_msa_baseline.CAN_CHANGE_LENGTH is False
    # ... and both of those answer a target length only through the SCAFFOLDED construction, which
    # is ours. The distinction is the point: a table must not print a scaffolded row as the paper's.
    assert evotune_baseline.LENGTH_CAPABILITY.target_length == SCAFFOLDED
    assert evodiff_msa_baseline.length_capability("inpaint").target_length == SCAFFOLDED
    # Their own entry point has no width to target, so it refuses rather than returning a
    # same-length row that a growth column would read as a failure.
    refusing = evodiff_msa_baseline.length_capability("unconditional")
    assert refusing.target_length == REFUSED
    with pytest.raises(LengthCapabilityError, match="unconditional"):
        refusing.check(240)

    # The names a table joins on.
    assert evotune_baseline.length_capability(False).method == "evotuned_plm"
    assert evotune_baseline.length_capability(True).method == "evotuned_plm_forced_substitutions"
    assert generation_eval.LENGTH_CAPABILITY.method == "edit_flows_model"

    # A self-contradictory declaration is refused at construction: "my kernel changes length" from a
    # method that cannot is the exact mistake a copy-pasted record makes.
    with pytest.raises(ValueError, match="can_change_length"):
        LengthCapability(method="x", can_change_length=False, target_length=NATIVE, mechanism="m" * 50)


def test_the_substitution_only_baselines_cannot_grow_and_the_declaration_says_so() -> None:
    """`CAN_CHANGE_LENGTH is False` for the evotuned baselines is a fact about the generator, checked.."""
    from editjumps.core.evotune.substitution import substitute_by_profile
    from editjumps.pipeline.evaluate import evodiff_msa_baseline, evotune_baseline

    propose = _stub_proposer()
    for length in (12, 40, 91):
        template = "".join(random.Random(length).choices("ACDEFGHIKLMNPQRSTVWY", k=length))
        weights = [1.0 / length] * length
        for budget in (0, 1, 4, length, length * 3):
            for forced in (False, True):
                for top_up in (False, True):
                    variant, _ = substitute_by_profile(
                        template, weights, budget, propose, random.Random(budget),
                        forced=forced, top_up=top_up,
                    )
                    assert len(variant) == len(template), (
                        "a substitution-only baseline produced a different length; if this can "
                        "happen, evotune_baseline.CAN_CHANGE_LENGTH is wrong"
                    )

    assert evotune_baseline.CAN_CHANGE_LENGTH is False
    # EvoDiff-MSA's default construction is fixed-width for the same reason one level up: every MSA row is.
    template = "QVQLVESGGGLVQPGGSLRLSCAAS"
    members = [template[:10] + "GGGG" + template[10:], template.replace("V", "I")]
    rows = evodiff_msa_baseline.build_alignment(template, members, msa_size=8, seed=0)
    assert {len(row) for row in rows} == {len(template)}, (
        "build_alignment must keep every row at the template's width; a wider row would mean the "
        "baseline could change length and the declaration is wrong"
    )
    assert evodiff_msa_baseline.CAN_CHANGE_LENGTH is False


def test_the_editor_can_grow_because_insert_is_an_event_of_its_ctmc() -> None:
    """`CAN_CHANGE_LENGTH is True` for the editor is a fact about the sampler, checked on both."""
    from editjumps.core.edit_flows.inference import euler_trace, gillespie_trace

    def insert_only(x: list[int], t: float) -> RateField:
        row = [0.0, 0.0, 0.0, 0.0, 1.0]  # every write emits token 4
        return [8.0] * len(x), [row] * len(x), [0.0] * len(x), [0.0] * len(x), [row] * len(x)

    x0 = [0, 1, 2, 3]
    for trace in (euler_trace, gillespie_trace):
        grew = False
        for seed in range(5):
            out, _ = trace(x0, insert_only, random.Random(seed), max_len=40)
            assert len(out) >= len(x0)
            grew = grew or len(out) > len(x0)
            assert len(out) <= 80, "the cap must bound growth, or a target above it is unbounded"
        assert grew, f"{trace.__name__} never inserted: the editor's growth claim would be false"


def test_a_growth_target_is_refused_where_it_has_no_construction() -> None:
    """The runners consult the declaration, so a refusal happens before any compute is spent."""
    from editjumps.core.length_capability import LengthCapabilityError
    from editjumps.pipeline.evaluate import evodiff_msa_baseline

    with pytest.raises(LengthCapabilityError, match="cannot be given a target length"):
        evodiff_msa_baseline.generate_variants(
            ["QVQLVESGGG"], ["QVQLVESGGG"], 2, 1, 0, Path("/tmp/never-written"),
            mode="unconditional", target_length=20,
        )
    # And the message says what to do instead, rather than only that it failed.
    reason = evodiff_msa_baseline.length_capability("unconditional").mechanism
    assert "inpaint" in reason and "inapplicable" in reason


