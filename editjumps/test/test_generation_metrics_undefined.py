"""Metrics the paper plots but never defines, kept visibly separate from the ones it does."""


def test_undefined_metrics_are_separated_from_defined_ones() -> None:
    """Metrics the paper plots but never defines must stay visibly separate from those it does."""
    import inspect

    from editjumps.core import generation_metrics, generation_metrics_undefined

    for name in ("entropy_delta", "js_divergence", "js_divergence_positional",
                 "profile_log_likelihood"):
        assert hasattr(generation_metrics_undefined, name), f"{name} belongs in the undefined module"
        assert not hasattr(generation_metrics, name), f"{name} must NOT sit beside the defined metrics"
        doc = inspect.getdoc(getattr(generation_metrics_undefined, name)) or ""
        assert "our reading" in doc.lower(), f"{name} must say plainly that it is our interpretation"

    # JS is bounded by ln 2 and symmetric; that much is forced by the definition we chose.
    import math

    js = generation_metrics_undefined.js_divergence(["AAAA"], ["WWWW"])
    assert 0.0 <= js <= math.log(2) + 1e-9
    assert abs(js - generation_metrics_undefined.js_divergence(["WWWW"], ["AAAA"])) < 1e-12
    assert generation_metrics_undefined.entropy_delta(["AC", "AC"], ["AC", "AD"]) < 0  # less diverse


def test_positional_js_is_sharper_than_the_composition_reading() -> None:
    """Per-column JS separates sets that share a composition; the pooled reading cannot."""
    import math

    from editjumps.core.generation_metrics_undefined import js_divergence, js_divergence_positional

    # Same multiset of residues at every position, mirrored: composition identical, positions not.
    left = ["AC", "CA"]
    right = ["AC", "AC"]
    pooled = js_divergence(left, right)
    positional = js_divergence_positional(left, right)
    assert pooled < 1e-9, f"pooled JS should be blind here, got {pooled}"
    assert positional > 0.05, f"positional JS should see it, got {positional}"

    # Bounds and symmetry.
    assert js_divergence_positional(left, left) < 1e-9
    assert abs(js_divergence_positional(left, right)
               - js_divergence_positional(right, left)) < 1e-12
    assert js_divergence_positional(["AAAA"], ["WWWW"]) <= math.log(2) + 1e-9
    assert js_divergence_positional([], ["AC"]) == 0.0
