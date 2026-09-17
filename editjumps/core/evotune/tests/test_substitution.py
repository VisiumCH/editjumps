"""The infill loop: what forcing changes, and what the budget top-up buys."""

import pytest


def test_evotune_forced_substitutions_are_what_separates_the_two_baselines() -> None:
    """The unforced baseline can spend its whole budget changing nothing; forcing removes that."""
    import random

    from editjumps.core.evotune.substitution import substitute_by_profile

    template = "ACDEFGHIKL"
    weights = [1 / len(template)] * len(template)

    def lazy(working: list[str | None], position: int, blocked: str) -> str:
        """Put the original residue back whenever that is not blocked."""
        original = template[position]
        return "W" if blocked == original else original

    # One round, no top-up: the unforced baseline masks its budget and mutates nothing.
    free, stats = substitute_by_profile(template, weights, 4, lazy, random.Random(0),
                                        forced=False, top_up=False)
    assert free == template
    assert stats["n_masked"] == 4 and stats["n_changed"] == 0 and stats["hit_budget"] is False

    forced, stats = substitute_by_profile(template, weights, 4, lazy, random.Random(0), forced=True)
    assert stats["n_changed"] == 4 and stats["hit_budget"] is True
    assert sum(a != b for a, b in zip(forced, template, strict=True)) == 4
    # Substitutions only, so length is invariant and no output needs realigning to its template.
    assert len(forced) == len(template)

def test_evotune_budget_top_up_hits_the_matched_mutation_count() -> None:
    """§4.2 matches MUTATIONS, not masks, so the unforced baseline has to keep drawing. ``top_up`` is our."""
    import random

    from editjumps.core.evotune.substitution import substitute_by_profile

    template = "A" * 20
    weights = [1 / 20] * 20

    def half_stubborn(working: list[str | None], position: int, blocked: str) -> str:
        """Change only the second half of the sequence; the first half is immovable."""
        return "W" if position >= 10 else "A"

    _, stats = substitute_by_profile(template, weights, 6, half_stubborn, random.Random(0),
                                     top_up=True, max_rounds=20)
    assert stats["n_changed"] == 6, "top-up must reach the matched budget"
    assert stats["n_masked"] > 6, "and it can only do so by masking more positions than the budget"

    # 10 mutable positions, 12 asked for: it must terminate and SAY it fell short.
    _, stats = substitute_by_profile(template, weights, 12, half_stubborn, random.Random(0),
                                     top_up=True, max_rounds=20)
    assert stats["n_changed"] == 10 and stats["hit_budget"] is False

    with pytest.raises(ValueError, match="ungapped template coordinates"):
        substitute_by_profile(template, [0.5, 0.5], 1, half_stubborn, random.Random(0))
