"""What each §4.2 method can do to a sequence's LENGTH — declared, so a benchmark can read it. §4.2's."""

from dataclasses import dataclass

#: The method's own transition kernel changes length.
NATIVE = "native"

#: Length is fixed by the method's construction, but the module can hand the model a width-``L`` problem — ``L.
SCAFFOLDED = "scaffolded"

#: No coherent construction exists, so a target length is an error.
REFUSED = "refused"

#: Valid values of :attr:`LengthCapability.target_length`.
TARGET_LENGTH_MODES: tuple[str, ...] = (NATIVE, SCAFFOLDED, REFUSED)


class LengthCapabilityError(ValueError):
    """Raised when a method is asked for a length it cannot coherently produce."""


@dataclass(frozen=True)
class LengthCapability:
    """One method's length capability, as audited from its source."""

    method: str
    can_change_length: bool
    target_length: str
    mechanism: str

    def __post_init__(self) -> None:
        """Validate the declaration."""
        if self.target_length not in TARGET_LENGTH_MODES:
            raise ValueError(
                f"{self.method}: target_length={self.target_length!r}; options: {TARGET_LENGTH_MODES}"
            )
        if self.target_length == NATIVE and not self.can_change_length:
            raise ValueError(
                f"{self.method}: target_length={NATIVE!r} claims the method's own kernel changes "
                f"length, but can_change_length is False. A method whose length is fixed by "
                f"construction is {SCAFFOLDED!r} at best."
            )
        if not self.mechanism.strip():
            raise ValueError(f"{self.method}: mechanism must say what makes growth possible or prevents it")

    def check(self, target_length: int | None) -> None:
        """Refuse a target length this method cannot coherently be given."""
        if target_length is None or self.target_length != REFUSED:
            return
        raise LengthCapabilityError(
            f"{self.method} cannot be given a target length: {self.mechanism} A benchmark should "
            f"record this row as inapplicable (see LengthCapability.inapplicable) rather than run "
            f"it and read a same-length output as a failure to grow."
        )

    def as_dict(self) -> dict[str, object]:
        """Render the declaration for a metrics report."""
        return {
            "method": self.method,
            "can_change_length": self.can_change_length,
            "target_length": self.target_length,
            "mechanism": self.mechanism,
        }

    def inapplicable(self, target_length: int) -> dict[str, object]:
        """Build the row a harness writes instead of a metric, when this method cannot play."""
        return {
            "method": self.method,
            "inapplicable": True,
            "requested_target_length": target_length,
            "reason": self.mechanism,
            "capability": self.as_dict(),
        }


def growth_summary(
    capability: LengthCapability,
    target_length: int,
    lengths_in: list[int],
    lengths_out: list[int],
) -> dict[str, object]:
    """Summarise what a run actually did to length, in the one shape every method reports it in."""
    if len(lengths_in) != len(lengths_out):
        raise ValueError(
            f"{len(lengths_in)} input lengths against {len(lengths_out)} output lengths; "
            "each generated sequence must be paired with the template it came from"
        )
    n = len(lengths_out)
    deltas = [out - into for into, out in zip(lengths_in, lengths_out, strict=True)]
    return {
        "target_length": target_length,
        "mode": capability.target_length,
        "capability": capability.as_dict(),
        "n_scored": n,
        "mean_length_in": (sum(lengths_in) / n) if n else 0.0,
        "mean_length_out": (sum(lengths_out) / n) if n else 0.0,
        "mean_delta": (sum(deltas) / n) if n else 0.0,
        "fraction_longer": (sum(1 for d in deltas if d > 0) / n) if n else 0.0,
        "fraction_at_target": (sum(1 for out in lengths_out if out >= target_length) / n) if n else 0.0,
    }
