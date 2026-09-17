"""The three-file run configuration behind ``editjumps evaluate`` — target / comparability / arm."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:  # import-cost-free: only for the annotation on ``ArmConfig.edit_flow_config``
    from editjumps.core.edit_flows.stages import EditFlowConfig

#: What a config field may hold once parsed.
FieldValue = str | int | float | bool | Path | None

#: Optional directory holding standalone YAML files; None means load from params.yaml.
DEFAULT_CONFIG_DIR: Path | None = None

#: Group name -> the file it is written to and read from.
CONFIG_FILENAMES: dict[str, str] = {
    "target": "target.yaml",
    "comparability": "comparability.yaml",
    "arm": "arm.yaml",
}

#: ``split.templates[0]`` — the first member of §4.2's inference part, which is the frame.
ALIGNMENT_FIRST_SPLIT_TEMPLATE = "first_split_template"

#: ``chosen[0][0]`` — the first sequence the evaluation's own permutation sampled.
ALIGNMENT_FIRST_SAMPLED = "first_sampled"

#: Valid values of :attr:`ComparabilityConfig.alignment_template`.
ALIGNMENT_TEMPLATES: tuple[str, ...] = (ALIGNMENT_FIRST_SPLIT_TEMPLATE, ALIGNMENT_FIRST_SAMPLED)

#: ``realised`` keeps drawing positions until the budget has actually mutated; ``masked`` masks that many once.
BUDGET_MODES: tuple[str, ...] = ("realised", "masked")

#: The quantities a cross-run comparison depends on, named as the *thing* rather than as a field, so that the.
COMPARABILITY_AXES: dict[str, tuple[str, ...]] = {
    "reference size (KL / MMD / agreement denominator)": ("holdout_size",),
    "alignment template (the per-position coordinate frame)": ("alignment_template", "n_templates"),
    "MMD sample size (generated side)": ("n_templates", "n_variants"),
    "agreement ceiling sample size": ("ceiling_n",),
    "which members land in templates / holdout / pool": ("split_seed",),
    "spectrum kernel": ("spectrum_k",),
    "matched mutation budget (§4.2)": ("mutations", "match_budget_to_editor", "budget_mode", "max_rounds"),
    "naturalness scorer (B.2 PLL)": ("pll_model", "pll_positions"),
}


def _coerce(raw: dict[str, Any], cls: type) -> dict[str, Any]:
    """Keep only the keys ``cls`` declares, converting to each field's annotated type."""
    typed: dict[str, Any] = {}
    for field in fields(cls):
        if field.name not in raw:
            continue
        value = raw[field.name]
        if value is None:
            typed[field.name] = None
            continue
        annotation = str(field.type)
        if "Path" in annotation:
            typed[field.name] = Path(str(value))
        elif "bool" in annotation:
            typed[field.name] = bool(value)
        elif "int" in annotation and "float" not in annotation:
            typed[field.name] = int(value)
        elif "float" in annotation:
            typed[field.name] = float(value)
        else:
            typed[field.name] = str(value)
    return typed


def _plain(value: FieldValue) -> str | int | float | bool | None:
    """Render one field value as something ``yaml.safe_dump``/``json.dumps`` accepts."""
    return str(value) if isinstance(value, Path) else value


@dataclass(frozen=True)
class TargetConfig:
    """What experiment is being run — the half of the settings two runs may legitimately differ on."""

    property: str | None = None
    seed_family: str | None = "Anti-SARS-CoV-2_VHH_Ty1.fasta"
    families_dir: Path = Path("data/interim/seed_families")
    pairs: Path = Path("data/pretrain/oas_homolog_pairs.tsv.gz")
    clock: float = 40.0

    def __post_init__(self) -> None:
        """Validate the target settings."""
        if self.property is not None:
            raise ValueError(
                f"target.property={self.property!r}: this repository reproduces EvoFlows, which is "
                "unconditional generation with no target property. Set `property: null`. See "
                "docs/claims.md, 'Why there is no target property'."
            )
        if self.clock < 0:
            raise ValueError(f"target.clock={self.clock}; must be >= 0 (0 = off)")

    # NOT a @property: this dataclass has a field literally named ``property``, which shadows the
    # builtin inside the class body, so the decorator is not available here.
    def family_fasta(self) -> Path | None:
        """Resolve `seed_family` against `families_dir`."""
        return None if self.seed_family is None else self.families_dir / self.seed_family


@dataclass(frozen=True)
class ComparabilityConfig:
    """The settings two runs must agree on before any of their numbers may be put side by side."""

    n_templates: int = 20
    n_variants: int = 20
    holdout_size: int = 200
    ceiling_n: int = 300
    alignment_template: str = ALIGNMENT_FIRST_SPLIT_TEMPLATE
    split_seed: int = 0
    spectrum_k: int = 3
    mutations: int = 4
    match_budget_to_editor: bool = True
    budget_mode: str = "realised"
    max_rounds: int = 8
    pll_model: str = ""
    pll_positions: int = 24

    def __post_init__(self) -> None:
        """Validate the enumerated fields and the counts that cannot be zero."""
        if self.alignment_template not in ALIGNMENT_TEMPLATES:
            raise ValueError(
                f"comparability.alignment_template={self.alignment_template!r} unknown; "
                f"options: {ALIGNMENT_TEMPLATES}"
            )
        if self.budget_mode not in BUDGET_MODES:
            raise ValueError(f"comparability.budget_mode={self.budget_mode!r}; options: {BUDGET_MODES}")
        for name in ("n_templates", "n_variants", "holdout_size", "ceiling_n", "spectrum_k", "mutations"):
            if getattr(self, name) < 1:
                raise ValueError(f"comparability.{name}={getattr(self, name)}; must be >= 1")

    @property
    def n_generated(self) -> int:
        """The generated-set size — the MMD sample size on the generated side."""
        return self.n_templates * self.n_variants

    def fingerprint(self) -> str:
        """Hash every comparability field into a short digest, for recording beside a metrics file."""
        canonical = json.dumps({f.name: _plain(getattr(self, f.name)) for f in fields(self)}, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:12]


@dataclass(frozen=True)
class ArmConfig:
    """The thing under test: one editor checkpoint, how it is parameterised and sampled."""

    model_folder: Path = Path("data/pretrain/edit_flows")
    rate_head: str = "linear"
    q_head: str = "fresh"
    path: str = "needleman_wunsch"
    schedule: str = "linear"
    loss: str = "edit_flow"
    sampler: str = "euler"
    n_steps: int = 50
    evotuned_model: Path = Path("data/pretrain/esm2_evotuned")
    evotune_train_corpus: Path = Path("data/pretrain/evotune_family.train.txt.gz")
    evotune_temperature: float = 1.0
    evotune_profile_size: int = 200
    evodiff_model: str = "msa-oa-dm-maxsub"
    evodiff_mode: str = "inpaint"
    evodiff_msa_size: int = 256
    evodiff_n_sequences: int = 64
    evodiff_selection_type: str = ""
    evodiff_temperature: float = 1.0
    evodiff_penalty_value: float = 2.0
    evodiff_device: str = "auto"

    def edit_flow_config(self) -> "EditFlowConfig":
        """Build the `~editjumps.core.edit_flows.stages.EditFlowConfig` this arm describes."""
        from editjumps.core.edit_flows.stages import EditFlowConfig

        return EditFlowConfig(
            path=self.path, schedule=self.schedule, loss=self.loss, sampler=self.sampler,
            rate_head=self.rate_head, q_head=self.q_head,
        )

    def __post_init__(self) -> None:
        """Validate every edit-head stage name through the shared registry."""
        self.edit_flow_config()


@dataclass(frozen=True)
class RunConfig:
    """The three groups together — one standard evaluation, fully described."""

    target: TargetConfig
    comparability: ComparabilityConfig
    arm: ArmConfig

    def validate(self) -> None:
        """Cross-check the three groups against each other."""
        has_family = self.target.seed_family is not None
        wants_split = self.comparability.alignment_template == ALIGNMENT_FIRST_SPLIT_TEMPLATE
        if wants_split and not has_family:
            raise ValueError(
                f"comparability.alignment_template={ALIGNMENT_FIRST_SPLIT_TEMPLATE!r} needs "
                "target.seed_family: without a family there is no §4.2 split, so `split.templates[0]` "
                f"does not exist. Set a seed family, or declare {ALIGNMENT_FIRST_SAMPLED!r} and accept "
                "that the frame moves with the seed and with n_templates."
            )
        if not wants_split and has_family:
            raise ValueError(
                f"comparability.alignment_template={ALIGNMENT_FIRST_SAMPLED!r} with a seed family: the "
                "baselines align to `split.templates[0]`, so the editor would sit in a different "
                "coordinate system from every method it is compared against. That is the alignment bug "
                f"of docs/findings.md. Use {ALIGNMENT_FIRST_SPLIT_TEMPLATE!r}."
            )

    @classmethod
    def from_params(cls, params: dict | None = None) -> "RunConfig":
        """Build the configuration equivalent to ``params.yaml`` plus the pipeline's CLI defaults."""
        from editjumps.core.utils import load_params

        params = load_params() if params is None else params
        edit_flows = params.get("edit_flows") or {}
        evotune = params.get("evotune") or {}
        evodiff = params.get("evodiff") or {}

        target = TargetConfig(
            # `str(None)` would produce the string "None" and trip the guard below; this
            # repository's only valid value is a real None.
            property=(lambda v: None if v is None else str(v))(params.get("property")),
            seed_family=str(evotune.get("family", TargetConfig.seed_family)),
            clock=float(edit_flows.get("clock", TargetConfig.clock)),
        )
        comparability = ComparabilityConfig(
            n_templates=int(evotune.get("n_templates", ComparabilityConfig.n_templates)),
            n_variants=int(evotune.get("n_variants", ComparabilityConfig.n_variants)),
            holdout_size=int(evotune.get("holdout_size", ComparabilityConfig.holdout_size)),
            split_seed=int(evotune.get("seed", ComparabilityConfig.split_seed)),
            spectrum_k=int(evotune.get("k", ComparabilityConfig.spectrum_k)),
            mutations=int(evotune.get("mutations", ComparabilityConfig.mutations)),
            budget_mode=str(evotune.get("budget_mode", ComparabilityConfig.budget_mode)),
            max_rounds=int(evotune.get("max_rounds", ComparabilityConfig.max_rounds)),
        )
        arm = ArmConfig(
            rate_head=str(edit_flows.get("rate_head", ArmConfig.rate_head)),
            q_head=str(edit_flows.get("q_head", ArmConfig.q_head)),
            path=str(edit_flows.get("path", ArmConfig.path)),
            schedule=str(edit_flows.get("schedule", ArmConfig.schedule)),
            loss=str(edit_flows.get("loss", ArmConfig.loss)),
            sampler=str(edit_flows.get("sampler", ArmConfig.sampler)),
            n_steps=int(edit_flows.get("eval_n_steps", ArmConfig.n_steps)),
            evotune_temperature=float(evotune.get("temperature", ArmConfig.evotune_temperature)),
            evotune_profile_size=int(evotune.get("profile_size", ArmConfig.evotune_profile_size)),
            evodiff_model=str(evodiff.get("model", ArmConfig.evodiff_model)),
            evodiff_mode=str(evodiff.get("mode", ArmConfig.evodiff_mode)),
            evodiff_msa_size=int(evodiff.get("msa_size", ArmConfig.evodiff_msa_size)),
            evodiff_n_sequences=int(evodiff.get("n_sequences", ArmConfig.evodiff_n_sequences)),
            evodiff_selection_type=str(evodiff.get("selection_type", ArmConfig.evodiff_selection_type)),
            evodiff_temperature=float(evodiff.get("temperature", ArmConfig.evodiff_temperature)),
            evodiff_penalty_value=float(evodiff.get("penalty_value", ArmConfig.evodiff_penalty_value)),
            evodiff_device=str(evodiff.get("device", ArmConfig.evodiff_device)),
        )
        config = cls(target=target, comparability=comparability, arm=arm)
        config.validate()
        return config

    @classmethod
    def load(
        cls,
        directory: Path | None = DEFAULT_CONFIG_DIR,
        *,
        params_file: Path | None = None,
        target: Path | None = None,
        comparability: Path | None = None,
        arm: Path | None = None,
    ) -> "RunConfig":
        """Load configuration from params.yaml or from target/comparability/arm files.

        If no directory or file overrides are given, loads directly from ``params.yaml``
        (or ``params_file``). If ``directory`` or individual file overrides are given,
        they override those parts of the configuration.
        """
        from editjumps.core.utils import load_params

        params = load_params(params_file) if params_file is not None else load_params()
        base = cls.from_params(params)

        if directory is None and target is None and comparability is None and arm is None:
            return base

        chosen = {"target": target, "comparability": comparability, "arm": arm}
        groups: dict[str, Any] = {}
        for name, klass in (("target", TargetConfig), ("comparability", ComparabilityConfig), ("arm", ArmConfig)):
            path = chosen[name]
            if path is not None:
                if not path.exists():
                    raise FileNotFoundError(f"{name} config not found: {path}")
                raw = yaml.safe_load(path.read_text()) or {}
                groups[name] = klass(**_coerce(raw, klass))
            elif directory is not None and (directory / CONFIG_FILENAMES[name]).exists():
                raw = yaml.safe_load((directory / CONFIG_FILENAMES[name]).read_text()) or {}
                groups[name] = klass(**_coerce(raw, klass))
            else:
                groups[name] = getattr(base, name)
        config = cls(**groups)
        config.validate()
        return config

    def write(self, directory: Path) -> dict[str, Path]:
        """Write the three groups to ``directory`` as YAML, one file per group."""
        directory.mkdir(parents=True, exist_ok=True)
        written: dict[str, Path] = {}
        for name in ("target", "comparability", "arm"):
            path = directory / CONFIG_FILENAMES[name]
            group = getattr(self, name)
            payload = {f.name: _plain(getattr(group, f.name)) for f in fields(group)}
            path.write_text(yaml.safe_dump(payload, sort_keys=False))
            written[name] = path
        return written

    def as_dict(self) -> dict[str, object]:
        """Render the whole configuration as plain JSON-able data, plus the comparability fingerprint."""
        out: dict[str, object] = {
            name: {f.name: _plain(getattr(getattr(self, name), f.name)) for f in fields(getattr(self, name))}
            for name in ("target", "comparability", "arm")
        }
        out["comparability_fingerprint"] = self.comparability.fingerprint()
        return out

    def replace(self, **overrides: TargetConfig | ComparabilityConfig | ArmConfig) -> "RunConfig":
        """Return a copy with whole groups replaced."""
        config = replace(self, **overrides)
        config.validate()
        return config

    # ---- the equivalent command lines ------------------------------------------------------- These exist so.

    def generation_eval_flags(self) -> dict[str, str]:
        """Build the flags equivalent to this config for ``editjumps generation-eval``."""
        family = self.target.family_fasta()
        flags = {
            "--model-folder": str(self.arm.model_folder),
            "--pairs": str(self.target.pairs),
            "--n-templates": str(self.comparability.n_templates),
            "--n-variants": str(self.comparability.n_variants),
            "--n-steps": str(self.arm.n_steps),
            "--holdout-size": str(self.comparability.holdout_size),
            "--ceiling-n": str(self.comparability.ceiling_n),
            "--seed": str(self.comparability.split_seed),
            "--k": str(self.comparability.spectrum_k),
            "--clock": str(self.target.clock),
            "--rate-head": self.arm.rate_head,
            "--q-head": self.arm.q_head,
            "--pll-model": self.comparability.pll_model,
            "--pll-positions": str(self.comparability.pll_positions),
        }
        if family is not None:
            flags["--family-fasta"] = str(family)
        return flags

    def evotune_baseline_flags(self) -> dict[str, str]:
        """Build the flags equivalent to this config for ``editjumps evotune-baseline``."""
        return {
            "--model-folder": str(self.arm.evotuned_model),
            "--family-fasta": str(self.target.family_fasta()),
            "--train-corpus": str(self.arm.evotune_train_corpus),
            "--n-templates": str(self.comparability.n_templates),
            "--n-variants": str(self.comparability.n_variants),
            "--mutations": str(self.comparability.mutations),
            "--temperature": str(self.arm.evotune_temperature),
            "--holdout-size": str(self.comparability.holdout_size),
            "--ceiling-n": str(self.comparability.ceiling_n),
            "--profile-size": str(self.arm.evotune_profile_size),
            "--budget-mode": self.comparability.budget_mode,
            "--max-rounds": str(self.comparability.max_rounds),
            "--seed": str(self.comparability.split_seed),
            "--k": str(self.comparability.spectrum_k),
        }

    def evodiff_baseline_flags(self) -> dict[str, str]:
        """Build the flags equivalent to this config for ``editjumps evodiff-msa-baseline``."""
        return {
            "--family-fasta": str(self.target.family_fasta()),
            "--mode": self.arm.evodiff_mode,
            "--model": self.arm.evodiff_model,
            "--n-templates": str(self.comparability.n_templates),
            "--n-variants": str(self.comparability.n_variants),
            "--mutations": str(self.comparability.mutations),
            "--holdout-size": str(self.comparability.holdout_size),
            "--ceiling-n": str(self.comparability.ceiling_n),
            "--msa-size": str(self.arm.evodiff_msa_size),
            "--n-sequences": str(self.arm.evodiff_n_sequences),
            "--selection-type": self.arm.evodiff_selection_type,
            "--temperature": str(self.arm.evodiff_temperature),
            "--penalty-value": str(self.arm.evodiff_penalty_value),
            "--device": self.arm.evodiff_device,
            "--budget-mode": self.comparability.budget_mode,
            "--max-rounds": str(self.comparability.max_rounds),
            "--seed": str(self.comparability.split_seed),
            "--k": str(self.comparability.spectrum_k),
        }
