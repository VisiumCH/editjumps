"""EvoFlows model + homolog-pair data loader (Phase 2b, model side)."""

# Annotations reference names imported only under TYPE_CHECKING below.
from __future__ import annotations

import copy
import gzip
import math
import random
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import torch
from torch import nn
from torch.utils.data import Dataset

from editjumps.core.edit_flows.inference import euler_trace, gillespie_trace
from editjumps.core.edit_flows.path import mixture_path, strip_epsilon
from editjumps.core.edit_flows.stages import EditFlowConfig
from editjumps.core.sequences import PAIR_SEP
from editjumps.pipeline.preprocess.pretrain.seed_homologs import CHAIN_CHOICES

if TYPE_CHECKING:
    from collections.abc import Sequence


def pick_device() -> str:
    """Best available torch device: CUDA (GPU boxes, e.g. the SkyPilot trainer), else MPS (Apple), else."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sinusoidal_embedding(t: float, dim: int) -> torch.Tensor:
    """Compute the standard sinusoidal embedding of a scalar time ``t`` in [0, 1]."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32) / half)
    args = t * freqs
    return torch.cat([torch.sin(args), torch.cos(args)])


#: Index into a ``VH<PAIR_SEP>VL`` split by ``--chain``. ``None`` preserves the line as-is.
_CHAIN_INDEX: dict[str, int | None] = {"heavy": 0, "light": 1, "joined": None}


def select_chain_pairs(
    pairs: "Sequence[tuple[str, str]]", chain: str = "joined"
) -> tuple[list[tuple[str, str]], dict[str, int]]:
    """Extract specified chain from paired sequences."""
    if chain not in CHAIN_CHOICES:
        raise ValueError(f"chain={chain!r}; options: {CHAIN_CHOICES}")
    skipped = {"no_separator": 0, "empty_chain": 0}
    if chain == "joined":
        return list(pairs), skipped
    index = _CHAIN_INDEX[chain]
    assert index is not None
    kept: list[tuple[str, str]] = []
    for x0, x1 in pairs:
        halves = [side.split(PAIR_SEP) for side in (x0, x1)]
        if any(len(h) < 2 for h in halves):
            skipped["no_separator"] += 1
            continue
        if index >= min(len(h) for h in halves):
            skipped["no_separator"] += 1
            continue
        chosen = (halves[0][index], halves[1][index])
        if not chosen[0] or not chosen[1]:
            skipped["empty_chain"] += 1
            continue
        kept.append(chosen)
    return kept, skipped


class HomologPairDataset(Dataset):
    """Homolog pairs -> Edit Flows training examples, via the config's ② path + schedule."""

    def __init__(self, pairs_path: Path, tokenizer: object, config: EditFlowConfig,
                 vocab_size: int, seed: int = 0, chain: str = "joined") -> None:
        """Load the homolog pairs and bind the tokenizer, stage config, and sampling RNG."""
        with gzip.open(pairs_path, "rt") as fh:
            raw = [tuple(line.rstrip("\n").split("\t")) for line in fh if "\t" in line]
        # Both members of every pair are split identically, and what the split drops is counted rather than.
        self.chain = chain
        self.pairs, self.skipped_pairs = select_chain_pairs(raw, chain)
        self.n_pairs_read = len(raw)
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        # Only the BLOSUM62 path needs the vocab (to score residue substitutions). `getattr` rather
        # than a required method, so a plain callable test double still constructs.
        get_vocab = getattr(tokenizer, "get_vocab", None)
        tokens = get_vocab() if callable(get_vocab) else None
        self._build_path = config.path_builder(tokens=tokens, vocab_size=vocab_size)
        self._schedule = config.schedule_fn()
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        """Return the number of homolog pairs."""
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        """Return one aligned, time-sampled training example."""
        x0, x1 = self.pairs[idx]
        ids0 = self.tokenizer(x0)["input_ids"]  # type: ignore[operator]
        ids1 = self.tokenizer(x1)["input_ids"]  # type: ignore[operator]
        z0, z1 = self._build_path(list(ids0), list(ids1), self.vocab_size, self.rng)
        t = self.rng.random()
        kappa, dkappa = self._schedule(t)
        z_t = mixture_path(z0, z1, kappa=kappa, rng=self.rng)
        return {"x_t": strip_epsilon(z_t), "z_t": z_t, "z_1": z1, "t": t,
                "kappa": kappa, "dkappa": dkappa}


class EvoFlowsModel(nn.Module):
    """ESM-2 trunk + FiLM time conditioning + shallow-MLP edit-rate heads (eqs 13-16). ``forward``."""

    def __init__(self, encoder: nn.Module, hidden_size: int, vocab_size: int, time_dim: int = 128,
                 lm_head: nn.Module | None = None, rate_head: str = "linear",
                 q_head: str = "fresh") -> None:
        """Wrap an ESM-style encoder with FiLM time conditioning and edit-rate heads."""
        if rate_head not in ("linear", "mlp"):
            raise ValueError(f"rate_head={rate_head!r}; options: linear, mlp")
        if q_head not in ("fresh", "esm_lm_head"):
            raise ValueError(f"q_head={q_head!r}; options: fresh, esm_lm_head")
        super().__init__()
        self.encoder = encoder
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size)
        )
        self.film = nn.Linear(hidden_size, 2 * hidden_size)

        def make_rate_head() -> nn.Module:
            if rate_head == "mlp":
                return nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.SiLU(),
                                     nn.Linear(hidden_size, 1))
            return nn.Linear(hidden_size, 1)

        self.rate_head, self.q_head = rate_head, q_head
        self.insert_lambda_head = make_rate_head()
        self.delete_lambda_head = make_rate_head()
        self.substitute_lambda_head = make_rate_head()
        if q_head == "esm_lm_head":
            if lm_head is None:
                raise ValueError("q_head='esm_lm_head' needs the trunk's lm_head; build via "
                                 "from_esm(), which loads EsmForMaskedLM rather than EsmModel")
            self.insert_q_head: nn.Module = copy.deepcopy(lm_head)
            self.substitute_q_head: nn.Module = copy.deepcopy(lm_head)
        else:
            self.insert_q_head = nn.Linear(hidden_size, vocab_size)
            self.substitute_q_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: torch.Tensor, t: float) -> tuple[torch.Tensor, ...]:
        """Predict per-position edit rates and token distributions for one sequence."""
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        h = self.encoder(input_ids=input_ids).last_hidden_state[0]  # (L, D)

        tau = self.time_mlp(sinusoidal_embedding(t, self.time_dim).to(h))  # (D,)
        gamma, beta = self.film(tau).chunk(2, dim=-1)
        h = h * (1.0 + gamma) + beta  # FiLM (eq 15)

        insert_lambda = nn.functional.softplus(self.insert_lambda_head(h)).squeeze(-1)
        delete_lambda = nn.functional.softplus(self.delete_lambda_head(h)).squeeze(-1)
        substitute_lambda = nn.functional.softplus(self.substitute_lambda_head(h)).squeeze(-1)
        insert_q = torch.softmax(self.insert_q_head(h), dim=-1)
        substitute_q = torch.softmax(self.substitute_q_head(h), dim=-1)
        return insert_lambda, insert_q, delete_lambda, substitute_lambda, substitute_q

    def forward_batch(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, times: "Sequence[float]"
    ) -> tuple[torch.Tensor, ...]:
        """Run the same computation as `forward`, over a padded batch of sequences. ``forward`` processes one."""
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state  # (B, L, D)

        # Stacked per-row rather than a vectorised sinusoidal_embedding, so the scalar path every
        # other caller uses is untouched. B is small; this is not hot.
        embedded = torch.stack([sinusoidal_embedding(float(t), self.time_dim) for t in times])
        tau = self.time_mlp(embedded.to(h))  # (B, D)
        gamma, beta = self.film(tau).chunk(2, dim=-1)
        h = h * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)  # FiLM (eq 15), per row

        insert_lambda = nn.functional.softplus(self.insert_lambda_head(h)).squeeze(-1)
        delete_lambda = nn.functional.softplus(self.delete_lambda_head(h)).squeeze(-1)
        substitute_lambda = nn.functional.softplus(self.substitute_lambda_head(h)).squeeze(-1)
        insert_q = torch.softmax(self.insert_q_head(h), dim=-1)
        substitute_q = torch.softmax(self.substitute_q_head(h), dim=-1)
        return insert_lambda, insert_q, delete_lambda, substitute_lambda, substitute_q

    @classmethod
    def from_esm(cls, model_name_or_path: str, time_dim: int = 128, rate_head: str = "linear",
                 q_head: str = "fresh") -> "EvoFlowsModel":
        """Build from a pretrained ESM-2 checkpoint (e.g. our MLM-continued trunk)."""
        from transformers import EsmForMaskedLM

        # EsmForMaskedLM rather than EsmModel: the latter drops `lm_head`, which eq 16 wants for the
        # Q heads, leaving Q randomly initialised.
        masked_lm = EsmForMaskedLM.from_pretrained(model_name_or_path)
        encoder = masked_lm.esm
        return cls(
            encoder,
            hidden_size=encoder.config.hidden_size,
            vocab_size=encoder.config.vocab_size,
            time_dim=time_dim,
            lm_head=masked_lm.lm_head,
            rate_head=rate_head,
            q_head=q_head,
        )

    @classmethod
    def load_trained(cls, output_folder: Path, time_dim: int = 128, rate_head: str = "linear",
                     q_head: str = "fresh") -> "EvoFlowsModel":
        """Reload a model saved by ``train_edit_flows`` (encoder dir + state dict)."""
        folder = Path(output_folder)
        if not (folder / "encoder").is_dir() or not (folder / "evoflows_model.pt").is_file():
            raise FileNotFoundError(
                f"{folder} is not a trained editor folder.\n\n"
                "`load_trained` needs `encoder/` and `evoflows_model.pt` side by side. "
                "See `editjumps restore-editor` to convert a training checkpoint."
            )
        model = cls.from_esm(str(folder / "encoder"), time_dim=time_dim,
                             rate_head=rate_head, q_head=q_head)
        state = torch.load(Path(output_folder) / "evoflows_model.pt", map_location="cpu")

        if "insert_lambda_head.0.weight" in state and model.rate_head != "mlp":
            raise RuntimeError(f"{output_folder} was trained with edit_flows.rate_head=mlp; "
                               f"set that in params.yaml to load it")
        if "insert_lambda_head.weight" in state and model.rate_head != "linear":
            raise RuntimeError(f"{output_folder} was trained with edit_flows.rate_head=linear; "
                               f"set that in params.yaml to load it")

        # Strip pooler keys if present from earlier checkpoint formats.
        pooler_keys = [k for k in state if k.startswith("encoder.pooler.")]
        for key in pooler_keys:
            del state[key]
        if pooler_keys:
            from editjumps.core.utils import get_logger

            get_logger(__file__).info(
                f"dropped {len(pooler_keys)} unused encoder-pooler tensor(s) from {output_folder}"
            )
        model.load_state_dict(state)
        return model


def pad_batch(
    examples: list[dict], pad_token_id: int, device: "torch.device | str"
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Pad a list of training examples into ``(input_ids, attention_mask, lengths)``."""
    lengths = [len(example["x_t"]) for example in examples]
    width = max(lengths)
    input_ids = torch.full((len(examples), width), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(examples), width), dtype=torch.long, device=device)
    for row, example in enumerate(examples):
        n = lengths[row]
        input_ids[row, :n] = torch.tensor(example["x_t"], dtype=torch.long, device=device)
        attention_mask[row, :n] = 1
    return input_ids, attention_mask, lengths


CHECKPOINT_NAME = "checkpoint.pt"


def restore_model_folder(
    checkpoint: Path,
    output_folder: Path,
    model_name: str,
    rate_head: str = "linear",
    q_head: str = "fresh",
    time_dim: int = 128,
) -> int:
    """Rebuild a loadable model folder from a training-state ``checkpoint.pt``."""
    from transformers import AutoTokenizer

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "model" not in state:
        raise KeyError(
            f"{checkpoint} has keys {sorted(state)[:6]} and no 'model' - it is not a training-state "
            f"checkpoint written by save_training_state"
        )
    model = EvoFlowsModel.from_esm(model_name, time_dim=time_dim, rate_head=rate_head, q_head=q_head)
    model.load_state_dict(state["model"])

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    # Same three writes as the end of train_edit_flows - including the tokenizer, without which the
    # folder is not self-contained and eval cannot reload it.
    model.encoder.save_pretrained(output_folder / "encoder")
    AutoTokenizer.from_pretrained(model_name).save_pretrained(output_folder / "encoder")
    torch.save(model.state_dict(), output_folder / "evoflows_model.pt")
    return int(state.get("step", -1))


class ResumeState(NamedTuple):
    """What a resumed run needs to continue as though it had never stopped.

    `step` alone is not enough: the best-weight tracker and the early-stopping counters live only in
    the loop's locals, so a checkpoint that omits them silently restarts patience from zero and
    throws away the best weights seen before the preemption.
    """

    step: int
    best_val: float = float("inf")
    best_step: int = -1
    best_state: dict | None = None
    patience_ref: float = float("inf")
    stale_evals: int = 0


def save_training_state(
    path: Path,
    step: int,
    model: nn.Module,
    optimizer: "torch.optim.Optimizer",
    best: ResumeState | None = None,
) -> None:
    """Save the FULL training state to a local ``checkpoint.pt`` so a run can resume exactly.

    Args:
        path: Where to write the checkpoint; parent directories are created.
        step: The step just completed.
        model: Model whose weights are saved.
        optimizer: Optimizer whose state is saved.
        best: The early-stopping bundle. It holds a second copy of the weights, so the file is
            roughly a third larger than model+optimizer alone - the price of not losing the best
            checkpoint to a preemption. Defaults to a fresh-run bundle.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    best = best or ResumeState(step=step)
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "python_rng": random.getstate(),
            "torch_rng": torch.get_rng_state(),
            "best_val": best.best_val,
            "best_step": best.best_step,
            "best_state": best.best_state,
            "patience_ref": best.patience_ref,
            "stale_evals": best.stale_evals,
        },
        path,
    )


def load_training_state(
    path: Path, model: nn.Module, optimizer: "torch.optim.Optimizer"
) -> ResumeState:
    """Load a ``checkpoint.pt`` written by `save_training_state` in place; return its resume state.

    Args:
        path: The checkpoint to read.
        model: Model to load the weights into, in place.
        optimizer: Optimizer to load the state into, in place.

    Returns:
        The step reached and the early-stopping bundle. Checkpoints written before that bundle
        existed lack those keys and fall back to the defaults a fresh run starts from.
    """
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    random.setstate(state["python_rng"])
    torch.set_rng_state(state["torch_rng"])
    return ResumeState(
        step=int(state["step"]),
        best_val=float(state.get("best_val", float("inf"))),
        best_step=int(state.get("best_step", -1)),
        best_state=state.get("best_state"),
        patience_ref=float(state.get("patience_ref", float("inf"))),
        stale_evals=int(state.get("stale_evals", 0)),
    )


def _rate_oracle(model: "EvoFlowsModel", device: "object",
                 observe: "object | None" = None) -> "object":
    """Wrap a torch model as the plain-list ``rate_fn(x, t)`` the pure ⑤ steppers consume."""

    def rate_fn(x: list[int], t: float) -> tuple:
        if observe is not None:
            observe(x, t)
        insert_lambda, insert_q, delete_lambda, substitute_lambda, substitute_q = model(
            torch.tensor(x, device=device), t)
        return (
            insert_lambda.tolist(), insert_q.tolist(), delete_lambda.tolist(),
            substitute_lambda.tolist(), substitute_q.tolist(),
        )

    return rate_fn


#: How many edits ``sample_edits_gillespie`` allows per ``n_steps``.
GILLESPIE_EVENTS_PER_STEP = 10


def sample_edits(
    model: "EvoFlowsModel",
    input_ids: list[int],
    n_steps: int = 50,
    rng: random.Random | None = None,
    max_len: int = 400,
    mask: list[bool] | None = None,
    clock: float | None = None,
    provenance: bool = False,
    observe: "object | None" = None,
) -> "list[int] | tuple[list[int], list[int | None]]":
    """⑤ Generate an edited token sequence from a starting sequence (Euler τ-leaping)."""
    rng = rng or random.Random(0)
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        x, prov = euler_trace(
            input_ids, _rate_oracle(model, device, observe), rng, n_steps=n_steps,
            max_len=max_len, mask=mask, clock=clock, track_provenance=provenance,
        )
    model.train()
    return (x, prov) if provenance else x


def sample_edits_gillespie(
    model: "EvoFlowsModel",
    input_ids: list[int],
    n_steps: int = 50,
    rng: random.Random | None = None,
    max_len: int = 400,
    mask: list[bool] | None = None,
    clock: float | None = None,
    provenance: bool = False,
    observe: "object | None" = None,
) -> "list[int] | tuple[list[int], list[int | None]]":
    """⑤ Generate by frozen-rate next-event (Gillespie/SSA) simulation — EvoFlows eqs 9-11, §3.3.."""
    rng = rng or random.Random(0)
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        x, prov = gillespie_trace(
            input_ids, _rate_oracle(model, device, observe), rng,
            max_events=GILLESPIE_EVENTS_PER_STEP * n_steps, max_len=max_len, mask=mask,
            clock=clock, track_provenance=provenance,
        )
    model.train()
    return (x, prov) if provenance else x


def train_edit_flows(
    pairs: Path,
    model_name: str = "data/pretrain/base_checkpoints/esm2_t12_35M_UR50D",
    output_folder: Path = Path("data/pretrain/edit_flows"),
    max_steps: int = 200,
    batch_size: int = 2,
    lr: float = 1e-4,
    grad_clip: float = 1.0,
    logging_steps: int = 10,
    save_steps: int = 100,
    seed: int = 0,
    config: EditFlowConfig | None = None,
    checkpoint_uri: str | None = None,
    val_frac: float = 0.05,
    eval_steps: int = 500,
    val_pairs: int = 256,
    patience: int = 0,
    min_delta: float = 0.0,
    run_tag: str | None = None,
    chain: str = "joined",
) -> Path:
    """Train the EvoFlows edit-flow model on homolog pairs and log to MLflow."""
    import random
    from statistics import median

    import mlflow
    import torch
    from transformers import AutoTokenizer

    from editjumps.core.cluster_split import split_pairs_by_family
    from editjumps.core.edit_flows.stages import resolve_loss
    from editjumps.core.gcs_checkpoints import gcs_download, gcs_upload
    from editjumps.core.utils import (
        design_space_tags,
        gcs_destination_tags,
        get_logger,
        start_mlflow_run,
    )

    logger = get_logger(__file__)
    config = config or EditFlowConfig()
    loss_fn = resolve_loss(config.loss)  # ④ dispatch the training objective by name
    device = pick_device()
    logger.info(f"device: {device}; edit-head stages: {config.as_dict()}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = HomologPairDataset(
        pairs, tokenizer, config, vocab_size=tokenizer.vocab_size, seed=seed, chain=chain
    )
    if len(dataset) == 0:
        raise ValueError(f"no homolog pairs in {pairs}; run `editjumps build-homolog-pairs` first")
    dropped = sum(dataset.skipped_pairs.values())
    logger.info(
        f"chain={chain}: {len(dataset)} of {dataset.n_pairs_read} pairs usable, {dropped} skipped "
        f"({dataset.skipped_pairs}); median length "
        f"{median([len(x0) for x0, _ in dataset.pairs]):.0f}"
    )
    train_idx, val_idx = split_pairs_by_family(dataset.pairs, val_frac, seed=seed)
    logger.info(f"{len(dataset)} homolog pairs -> {len(train_idx)} train / {len(val_idx)} val "
                f"(sequence-disjoint, val_frac={val_frac}); loading trunk {model_name}")
    if val_frac > 0 and not val_idx:
        logger.warning("val_frac > 0 but the split produced no held-out pairs - no val loss "
                       "will be logged; check that the pairs file has more than one family")

    model = EvoFlowsModel.from_esm(
        model_name, rate_head=config.rate_head, q_head=config.q_head
    ).to(device)
    model.train()
    # ESM derives position ids from the pad id, so batching must pad with the tokenizer's own.
    # Falling back to the encoder config keeps a tokenizer without an explicit pad token working.
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = model.encoder.config.pad_token_id
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # The global (seeded) `random` stream, so pair sampling's position is captured in the
    # checkpoint's python-RNG state and restored on resume.
    random.seed(seed)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Resume-from-latest. Any GCS failure — including a fresh run with nothing there yet — falls
    # back to a fresh start.
    local_checkpoint = output_folder / CHECKPOINT_NAME
    start_step = 0
    resumed = ResumeState(step=-1)
    if checkpoint_uri:
        remote_checkpoint = f"{checkpoint_uri.rstrip('/')}/{CHECKPOINT_NAME}"
        if gcs_download(remote_checkpoint, local_checkpoint):
            try:
                resumed = load_training_state(local_checkpoint, model, optimizer)
                start_step = resumed.step + 1
                logger.info(f"resuming from step {start_step} (checkpoint {remote_checkpoint})")
                if resumed.best_step >= 0:
                    logger.info(f"carried over best val_loss {resumed.best_val:.3f} from step "
                                f"{resumed.best_step} ({resumed.stale_evals} stale evaluations)")
            except Exception as exc:  # noqa: BLE001 - a corrupt/partial checkpoint must not crash the run
                logger.warning(f"failed to load {local_checkpoint}: {type(exc).__name__}: {exc}; starting fresh")
                start_step, resumed = 0, ResumeState(step=-1)

    tags = design_space_tags(
        objective="edit_flows",
        backbone=Path(model_name).name,
        weight_init="pretrained",
        pretrain_data=pairs.name,
        # One chain per example IS the "separate" regime, and evotune (single-chain family FASTA)
        # cannot disagree about what a run was fed.
        chain_handling="separate" if chain != "joined" else "combine",
    )
    run_name = f"editflows-{Path(model_name).name}" + (f"-{run_tag}" if run_tag else "")
    if run_tag:
        tags = {**tags, "run_tag": run_tag}
    # The trainer is handed --checkpoint-uri and is the only place that knows it, so it goes on
    # the run explicitly; the model and metrics destinations come from the job environment.
    tags = {**tags, **gcs_destination_tags(gcs_checkpoint_uri=checkpoint_uri)}
    with start_mlflow_run("editjumps-edit-flows", run_name=run_name, tags=tags):
        mlflow.log_params(
            {
                "model_name": model_name,
                "n_pairs": len(dataset),
                "chain": chain,
                "n_pairs_read": dataset.n_pairs_read,
                "n_pairs_skipped": dropped,
                "n_train_pairs": len(train_idx),
                "n_val_pairs": len(val_idx),
                "val_frac": val_frac,
                "eval_steps": eval_steps,
                "max_steps": max_steps,
                "batch_size": batch_size,
                "lr": lr,
                "grad_clip": grad_clip,
                **config.as_dict(),  # edit-head stage choices: path/schedule/loss/sampler
            }
        )
        def write_checkpoint(step: int) -> None:
            """Save the full training state locally + mirror to GCS."""
            if not checkpoint_uri:
                return
            # The early-stopping locals are assigned below; every call site runs after that, so the
            # closure reads the live values rather than a stale copy.
            save_training_state(
                local_checkpoint, step, model, optimizer,
                best=ResumeState(step, best_val, best_step, best_state, patience_ref, stale_evals),
            )
            gcs_upload(local_checkpoint, f"{checkpoint_uri.rstrip('/')}/{CHECKPOINT_NAME}")

        # Sample the val examples ONCE. `__getitem__` redraws t and the mixture path per call, so
        # re-sampling per evaluation would make the curve sampling noise rather than learning.
        val_examples: list[dict] = []
        if val_idx:
            picker = random.Random(seed + 1)
            for i in picker.sample(val_idx, min(val_pairs, len(val_idx))):
                example = dataset[i]
                if example["x_t"]:
                    val_examples.append(example)
            logger.info(f"validation: {len(val_examples)} fixed held-out examples every {eval_steps} steps")

        def val_loss() -> float:
            """Mean edit-flow loss over the fixed held-out examples, with no gradient."""
            model.eval()
            total = 0.0
            with torch.no_grad():
                for example in val_examples:
                    rates = model(torch.tensor(example["x_t"], device=device), example["t"])
                    total += loss_fn(
                        example["z_t"], example["z_1"], example["kappa"], example["dkappa"], *rates
                    ).item()
            model.train()
            return total / max(len(val_examples), 1)

        losses: list[float] = []
        val_history: list[tuple[int, float]] = []
        # `best_*` tracks the TRUE minimum and decides which weights get shipped (held on CPU so the copy does.
        best_val, best_step, best_state = resumed.best_val, resumed.best_step, resumed.best_state
        diverged = False
        patience_ref, stale_evals = resumed.patience_ref, resumed.stale_evals
        stopped_early = False
        # A resume at or past max_steps enters no iteration, and the code below reads `step`.
        step = start_step - 1
        for step in range(start_step, max_steps):
            optimizer.zero_grad()
            batch_loss = torch.zeros((), device=device)
            # Draw first, forward once. The draw loop makes the same calls in the same order as the
            # accumulating version, so the RNG stream (and a resumed run) is identical.
            examples = []
            for _ in range(batch_size):
                example = dataset[train_idx[random.randrange(len(train_idx))]]
                if example["x_t"]:
                    examples.append(example)
            if examples:
                input_ids, attention_mask, lengths = pad_batch(examples, pad_token_id, device)
                rates = model.forward_batch(input_ids, attention_mask, [e["t"] for e in examples])
                for row, (example, n) in enumerate(zip(examples, lengths, strict=True)):
                    # Slice each row back to its true length and reuse the per-example loss unchanged.
                    batch_loss = batch_loss + loss_fn(
                        example["z_t"], example["z_1"], example["kappa"], example["dkappa"],
                        *(rate[row, :n] for rate in rates),
                    )
            # Divided by the REQUESTED batch_size, not len(examples): changing that would silently
            # rescale the learning rate relative to every run on record.
            batch_loss = batch_loss / batch_size
            batch_loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip))

            # Check BEFORE stepping: a step on a non-finite loss or gradient poisons the Adam
            # moments, which restoring the best weights does not undo.
            loss_value = batch_loss.item()
            if not math.isfinite(loss_value) or not math.isfinite(grad_norm):
                diverged = True
                logger.error(f"loss became {loss_value} (grad norm {grad_norm}) at step {step} - the "
                             f"run has diverged; stopping before the optimizer step so the weights "
                             f"and the optimizer state stay finite. Lower the learning rate "
                             f"(currently {lr}).")
                break

            optimizer.step()
            losses.append(loss_value)
            if step % logging_steps == 0:
                mlflow.log_metric("loss", losses[-1], step=step)
                logger.info(f"step {step}/{max_steps} loss {losses[-1]:.3f}")
            if val_examples and eval_steps and step % eval_steps == 0:
                current = val_loss()
                val_history.append((step, current))
                mlflow.log_metric("val_loss", current, step=step)
                # `loss` is ONE batch and val_loss averages `val_pairs` examples, so comparing them directly.
                recent = losses[-eval_steps:] or losses
                train_mean = sum(recent) / len(recent)
                mlflow.log_metric("train_loss_mean", train_mean, step=step)
                logger.info(f"step {step}/{max_steps} val_loss {current:.3f} "
                            f"train_loss_mean {train_mean:.3f} (last batch {losses[-1]:.3f})")

                if current < best_val:
                    best_val, best_step = current, step
                    best_state = {k: v.detach().to("cpu").clone() for k, v in model.state_dict().items()}
                if current < patience_ref - min_delta:
                    patience_ref, stale_evals = current, 0
                else:
                    stale_evals += 1
                    if patience and stale_evals >= patience:
                        stopped_early = True
                        # Says only what happened; the restore is decided and logged below.
                        logger.info(
                            f"early stop at step {step}: {stale_evals} validations without a "
                            f">{min_delta} improvement (best val_loss {best_val:.3f} at step "
                            f"{best_step})"
                        )
                        break
            if save_steps and step > 0 and step % save_steps == 0:
                model.encoder.save_pretrained(output_folder / "encoder")
                write_checkpoint(step)  # full-state checkpoint -> GCS for preemption resilience

        # Order matters: measure the FINAL weights, then decide whether the best earlier weights beat them.
        final_val = None
        if val_examples:
            final_val = val_loss()
            val_history.append((step, final_val))
            mlflow.log_metric("val_loss", final_val, step=step)
            logger.info(f"final val_loss {final_val:.3f} at step {step}")

        # Ship the best weights whether or not patience fired: a run that reaches max_steps past its own val.
        final_broken = final_val is None or not math.isfinite(final_val)
        restored_from = None
        if best_state is not None and (final_broken or best_val < final_val):
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
            restored_from = best_step
            reason = "the final weights are not finite" if final_broken else f"val_loss {final_val:.3f}"
            logger.info(f"restored the step-{best_step} weights (val_loss {best_val:.3f}) over the "
                        f"step-{step} ones ({reason})")
        elif final_broken:
            logger.error("the run diverged and there is no finite checkpoint to fall back to - "
                         "the saved weights are unusable; lower the learning rate and re-run")

        write_checkpoint(step)  # final full-state checkpoint (resume is a no-op past this)
        model.encoder.save_pretrained(output_folder / "encoder")
        tokenizer.save_pretrained(output_folder / "encoder")  # self-contained: eval reloads tokenizer from here
        torch.save(model.state_dict(), output_folder / "evoflows_model.pt")

        mlflow.log_metric("stopped_early", float(stopped_early))
        mlflow.log_metric("diverged", float(diverged))
        mlflow.log_metric("stopped_at_step", float(step))
        mlflow.log_metric("saved_weights_from_step", float(restored_from if restored_from is not None else step))
        # Loss is noisy (the κ̇/(1−κ) weight spikes near t=1), so the headline is a median over the
        # first vs last window. (A resumed run already past max_steps does no steps -> skip.)
        if losses:
            window = max(1, min(20, len(losses) // 3))
            first, last = median(losses[:window]), median(losses[-window:])
            mlflow.log_metric("loss_first_window_median", first)
            mlflow.log_metric("loss_last_window_median", last)
            logger.info(f"median loss: first {window} steps {first:.3f} -> last {window} steps {last:.3f}")
        if val_history:
            best_at, best = min(val_history, key=lambda kv: kv[1])
            mlflow.log_metric("val_loss_final", val_history[-1][1])
            mlflow.log_metric("val_loss_best", best)
            mlflow.log_metric("val_loss_best_step", best_at)
            if stopped_early:
                logger.info(f"stopped early at step {step} of {max_steps}; best val_loss {best:.3f} "
                            f"at step {best_at}, and those are the saved weights")
            elif val_history[-1][1] > best:
                logger.warning(f"ran the full {max_steps} steps but val_loss peaked at {best:.3f} "
                               f"(step {best_at}) and ended at {val_history[-1][1]:.3f} - the tail was "
                               f"overfitting; consider a smaller patience next run")
            else:
                logger.info(f"val_loss {best:.3f} at step {best_at} (its best, and the final value)")
    logger.info(f"saved EvoFlows model to {output_folder}")
    return output_folder
