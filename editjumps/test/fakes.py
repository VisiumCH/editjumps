"""Fakes and stubs shared by more than one test module."""

import random
from pathlib import Path

import pytest

from editjumps.core.evotune.substitution import Proposer

# --- EvoDiff-MSA: §4.2's fifth baseline, and the seam to somebody else's environment --------- The real model.
FAKE_EVODIFF_RUNNER = '''
import json, os, sys

FAVOURITE = "W"          # logit 0.0; every other residue -30, so the draw is W unless W is blocked
PROPOSABLE = "ACDEFGHIKLMNPQRSTVWY"
log = open(os.environ["FAKE_EVODIFF_LOG"], "a")
state = {"length": 0, "rows": 0, "path": "", "query": ""}

def handle(req):
    op = req.get("op")
    if op == "ping":
        return {"model": "fake", "device": "cpu", "scheme": "mask"}
    if op == "msa":
        rows = [l.strip() for l in open(req["path"]) if not l.startswith(">") and l.strip()]
        if len(rows) < int(req["n_sequences"]):
            return {"error": "msa num_seqs < n_sequences"}
        state.update(length=len(rows[0]), rows=int(req["n_sequences"]), path=req["path"],
                     query=rows[0])
        return {"n_sequences": state["rows"], "length": state["length"],
                "query_length": len(rows[0]), "query": rows[0]}
    if op == "fill":
        if not state["length"]:
            return {"error": "no MSA loaded"}
        query, position = req["query"], int(req["position"])
        if len(query) != state["length"]:
            return {"error": "query is %d, alignment is %d" % (len(query), state["length"])}
        if not 0 <= position < state["length"]:
            return {"error": "position %d out of range" % position}
        if query[position] != "#":
            return {"error": "asked to fill an unmasked position %d" % position}
        log.write(json.dumps({"op": "fill", "position": position,
                              "masks": query.count("#"), "query": query}) + "\\n")
        log.flush()
        return {"logits": {r: (0.0 if r == FAVOURITE else -30.0) for r in PROPOSABLE}}
    if op == "generate_query":
        log.write(json.dumps({"op": "generate_query"}) + "\\n")
        log.flush()
        return {"query": FAVOURITE * state["length"]}
    return {"error": "unknown op %r" % op}

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    req = json.loads(line)
    op = req.get("op")
    if op == "close":
        break
    print(json.dumps(handle(req)), flush=True)
'''


def _fake_shim(folder: Path, body: str = FAKE_EVODIFF_RUNNER) -> tuple[str, Path]:
    """Write an executable stand-in for `.venv/bin/evodiff-msa`, plus its request log."""
    import os
    import sys

    folder.mkdir(parents=True, exist_ok=True)
    script = folder / "fake_runner.py"
    script.write_text(body)
    log = folder / "requests.jsonl"
    shim = folder / "evodiff-msa"
    shim.write_text(f'#!/bin/sh\nFAKE_EVODIFF_LOG="{log}" exec "{sys.executable}" "{script}" "$@"\n')
    os.chmod(shim, 0o755)
    return str(shim), log


def _stub_the_gpu_half_of_generation_eval(monkeypatch: "pytest.MonkeyPatch") -> None:
    """Replace the three pieces of `generation_eval.evaluate` that need a checkpoint."""
    import sys
    import types

    alphabet = "ACDEFGHIKLMNPQRSTVWY"

    class _Tokenizer:
        """Character-level stand-in: one token per residue, so decode(encode(s)) == s."""

        def __call__(self, text: str) -> dict:
            return {"input_ids": [ord(c) for c in text]}

        def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
            # Spaces because the caller strips them, matching the real tokenizer's output.
            return " ".join(chr(i) for i in ids)

    class _TokenizerFactory:
        @staticmethod
        def from_pretrained(path: str) -> "_Tokenizer":
            return _Tokenizer()

    class _Model:
        def to(self, device: object) -> "_Model":
            return self

        def eval(self) -> "_Model":
            return self

        @classmethod
        def load_trained(cls, folder: object, rate_head: str = "linear",
                         q_head: str = "fresh") -> "_Model":
            return cls()

    def _sample_edits(model: object, ids: list[int], n_steps: int = 50,
                      clock: float | None = None,
                      rng: "random.Random | None" = None) -> list[int]:
        rng = rng or random.Random(0)
        out = list(ids)
        for _ in range(5):
            position = rng.randrange(len(out))
            kind = rng.choice(("sub", "ins", "del"))
            if kind == "sub":
                out[position] = ord(rng.choice(alphabet))
            elif kind == "ins":
                out.insert(position, ord(rng.choice(alphabet)))
            else:
                out.pop(position)
        return out

    # Populated through `__dict__`, not attribute assignment: a bare `ModuleType` has no declared
    # attributes, so `ty` rejects `fake.AutoTokenizer = ...` as unresolved.
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.__dict__["AutoTokenizer"] = _TokenizerFactory
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    fake_evoflows = types.ModuleType("editjumps.pipeline.train.evoflows")
    fake_evoflows.__dict__.update({
        "EvoFlowsModel": _Model,
        "pick_device": lambda: "cpu",
        "sample_edits": _sample_edits,
    })
    monkeypatch.setitem(sys.modules, "editjumps.pipeline.train.evoflows", fake_evoflows)


def _stub_proposer(favourite: str = "W") -> "Proposer":
    """Build a torch-free `Proposer` that always answers one residue, or its neighbour if blocked."""
    def propose(working: list[str | None], position: int, blocked: str) -> str:
        return "A" if blocked == favourite else favourite

    return propose
