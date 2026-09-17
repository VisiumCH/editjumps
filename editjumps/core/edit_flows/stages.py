"""The edit-head pipeline as swappable stages — a 1:1 map of this folder's README DAG."""

import random
from dataclasses import dataclass, replace
from typing import Callable

from editjumps.core.edit_flows.alignment import blosum62_scoring, needleman_wunsch, needleman_wunsch_affine
from editjumps.core.edit_flows.path import sample_edit_path

# --- ② path: raw (ids0, ids1) -> aligned gapped pair (z0, z1) -----------------
# A path builder maps two token-id sequences (+ vocab size, rng) to the aligned pair.
PathBuilder = Callable[[list[int], list[int], int, random.Random], tuple[list[int], list[int]]]


def _path_needleman_wunsch(ids0: list[int], ids1: list[int], vocab_size: int,
                           rng: random.Random) -> tuple[list[int], list[int]]:
    """EvoFlow path: globally align two *real* homolog token sequences (``ids0`` -> ``ids1``)."""
    return needleman_wunsch(ids0, ids1)


def _path_random_noise(ids0: list[int], ids1: list[int], vocab_size: int,
                       rng: random.Random) -> tuple[list[int], list[int]]:
    """Edit Flows path: edit *random noise* into the data ``ids1`` (``ids0`` ignored)."""
    bos, body = ids1[0], ids1[1:]
    n = len(body)
    n_sub = rng.randint(0, n)
    n_del = rng.randint(0, max(1, n // 2))
    z0, z1 = sample_edit_path(body, vocab_size, target_n_sub=n_sub, target_n_del=n_del, rng=rng)
    return [bos] + z0, [bos] + z1


def _path_needleman_wunsch_blosum62(ids0: list[int], ids1: list[int], vocab_size: int,
                                    rng: random.Random) -> tuple[list[int], list[int]]:
    """Refuse the BLOSUM62 path, which cannot be built from this signature alone."""
    raise ValueError(
        "edit_flows.path='needleman_wunsch_blosum62' needs the tokenizer vocab to score residue "
        "substitutions; build it with EditFlowConfig.path_builder(tokens=tokenizer.get_vocab(), "
        f"vocab_size={vocab_size})"
    )


PATH_BUILDERS: dict[str, PathBuilder] = {
    "needleman_wunsch": _path_needleman_wunsch,  # EvoFlow: homolog <-> homolog, unit costs (ours)
    # Same alignment under the scoring EvoFlow's §3.2 citation implies (affine gaps + BLOSUM62), and.
    "needleman_wunsch_blosum62": _path_needleman_wunsch_blosum62,
    "random_noise": _path_random_noise,          # Edit Flows: data <-> random noise
}

# --- ② schedule: time t -> (kappa, dkappa) ------------------------------------
SCHEDULES: dict[str, Callable[[float], tuple[float, float]]] = {
    "linear": lambda t: (t, 1.0),  # Meta's reference schedule (kappa = t)
    # UNSPECIFIED 1.
    "cubic": lambda t: (t**3, 3.0 * t**2),  # Edit Flows' schedule (kappa = t^3)
    # extension point: add e.g. "cosine" here and it becomes selectable.
}

# --- ④ loss / ⑤ sampler: torch-backed, resolved lazily by name ---------------
LOSSES: tuple[str, ...] = ("edit_flow", "edit_flow_figure13")
SAMPLERS: tuple[str, ...] = ("euler", "gillespie")


def resolve_loss(name: str) -> Callable:
    """Return the ④ loss function for ``name`` (lazy torch import); extend by adding a branch.."""
    if name == "edit_flow":
        from editjumps.core.edit_flows.loss import edit_flow_loss
        return edit_flow_loss
    if name == "edit_flow_figure13":
        import functools

        from editjumps.core.edit_flows.loss import edit_flow_loss
        return functools.partial(edit_flow_loss, supervision="figure13")
    raise NotImplementedError(f"loss {name!r} not implemented; options: {LOSSES}")


def resolve_sampler(name: str) -> Callable:
    """Return the ⑤ sampler function for ``name`` (lazy torch import)."""
    if name == "euler":
        from editjumps.pipeline.train.evoflows import sample_edits
        return sample_edits
    if name == "gillespie":
        from editjumps.pipeline.train.evoflows import sample_edits_gillespie
        return sample_edits_gillespie
    raise NotImplementedError(f"sampler {name!r} not implemented; options: {SAMPLERS}")


#: ③ How each per-position rate head is parameterised. ``mlp`` is EvoFlow Appendix A's "shallow MLPs".
RATE_HEADS: tuple[str, ...] = ("linear", "mlp")

#: ③ How each token distribution Q is parameterised. ``esm_lm_head`` copies the trunk's pretrained masked-LM.
Q_HEADS: tuple[str, ...] = ("fresh", "esm_lm_head")


@dataclass(frozen=True)
class EditFlowConfig:
    """One choice per DAG stage — the swappable edit-head configuration."""

    path: str = "needleman_wunsch"
    schedule: str = "linear"
    loss: str = "edit_flow"
    sampler: str = "euler"
    # ③ head parameterisation. Defaults are what existing checkpoints were trained with; the other
    # option in each case is EvoFlow Appendix A's. Both change parameter shapes — see params.yaml.
    rate_head: str = "linear"    # linear | mlp
    q_head: str = "fresh"        # fresh | esm_lm_head

    def __post_init__(self) -> None:
        """Validate every stage name against its registry (fail fast on a typo)."""
        for field, name, valid in [
            ("path", self.path, PATH_BUILDERS), ("schedule", self.schedule, SCHEDULES),
            ("loss", self.loss, LOSSES), ("sampler", self.sampler, SAMPLERS),
            ("rate_head", self.rate_head, RATE_HEADS), ("q_head", self.q_head, Q_HEADS),
        ]:
            if name not in valid:
                raise ValueError(f"edit_flows.{field}={name!r} unknown; options: {tuple(valid)}")

    @classmethod
    def from_params(cls, params: dict, **overrides: str | float | None) -> "EditFlowConfig":
        """Build from a parsed ``params.yaml`` dict's ``edit_flows`` block; ``None`` overrides are ignored."""
        # Derived from the dataclass rather than hand-listed: a hand-written allow-list silently dropped.
        keys = set(cls.__dataclass_fields__)
        block = {k: v for k, v in (params or {}).get("edit_flows", {}).items() if k in keys}
        cfg = cls(**block)
        applied = {k: v for k, v in overrides.items() if v is not None}
        return replace(cfg, **applied) if applied else cfg

    def as_dict(self) -> dict[str, str | float]:
        """Stage -> chosen name/value (for ``mlflow.log_params``)."""
        # Every field, for the same reason: a hand-listed subset means a new knob is chosen but
        # never logged, so an MLflow run cannot be told apart from one that used the default.
        return {f: getattr(self, f) for f in self.__dataclass_fields__}

    def path_builder(self, tokens: dict[str, int] | None = None,
                     vocab_size: int | None = None) -> Callable:
        """Return this config's ② path builder, configured for the tokenizer if it needs one.."""
        if self.path == "needleman_wunsch_blosum62":
            if not tokens or not vocab_size:
                return PATH_BUILDERS[self.path]  # raises with the actionable message
            scoring = blosum62_scoring(tokens, vocab_size)

            def build(ids0: list[int], ids1: list[int], _vocab_size: int,
                      _rng: random.Random) -> tuple[list[int], list[int]]:
                """Align two homolog token sequences under affine gaps + BLOSUM62, symmetrically."""
                return needleman_wunsch_affine(ids0, ids1, scoring)

            return build
        return PATH_BUILDERS[self.path]

    def schedule_fn(self) -> Callable[[float], tuple[float, float]]:
        """Return this config's ② schedule ``t -> (kappa, dkappa)``."""
        return SCHEDULES[self.schedule]
