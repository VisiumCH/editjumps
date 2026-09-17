"""The seam to somebody else's environment: a live EvoDiff-MSA process behind one callable.."""

import json
import random
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path
from types import TracebackType
from typing import Any

from editjumps.core.evodiff_msa.alignment import GAP
from editjumps.core.evodiff_msa.sampling import softmax_choice

#: Mask character on the wire.
MASK_CHAR = "#"

#: The shim ``editjumps/core/evodiff_msa/install_evodiff.sh`` puts on ``.venv/bin``. Resolved via ``PATH``, exactly
SHIM = "evodiff-msa"

#: UNSPECIFIED 1.
MODELS: dict[str, str] = {
    "msa-oa-dm-maxsub": "MaxHamming",
    "msa-oa-dm-randsub": "random",
}

#: The wire protocol between this module and the shim, in one place so the client, the runner and the tests'.
PROTOCOL: tuple[str, ...] = ("msa", "fill", "generate_query", "ping", "close")


def row_from_working(working: Sequence[str | None], length: int) -> str:
    """Render ``substitute_by_profile``'s working residue list as an MSA row, ``None`` -> `MASK_CHAR`.."""
    if len(working) != length:
        raise ValueError(
            f"working row is {len(working)} but the alignment is {length} columns wide; "
            "the query must stay in the alignment's coordinates"
        )
    return "".join(MASK_CHAR if residue is None else residue for residue in working)


class MsaProposer:
    """The network half of the baseline: a live EvoDiff-MSA process behind the ``Proposer`` protocol."""

    def __init__(
        self,
        a3m: Path,
        seed: int,
        *,
        n_sequences: int = 64,
        model: str = "msa-oa-dm-maxsub",
        selection_type: str = "",
        max_seq_len: int = 256,
        temperature: float = 1.0,
        device: str = "auto",
        shim: str = SHIM,
        expect_query: str = "",
    ) -> None:
        """Validate the configuration; the process itself starts in ``__enter__``. ``seed`` seeds the residue."""
        if model not in MODELS:
            raise ValueError(f"model={model!r}; options: {sorted(MODELS)}")
        selection = selection_type or MODELS[model]
        if selection not in set(MODELS.values()):
            raise ValueError(f"selection_type={selection!r}; options: {sorted(set(MODELS.values()))}")
        if n_sequences < 2:
            raise ValueError(f"n_sequences={n_sequences}; an MSA of one row is just the query")
        self.a3m = a3m
        self.seed = seed
        self.n_sequences = n_sequences
        self.model = model
        self.selection_type = selection
        self.max_seq_len = max_seq_len
        self.temperature = temperature
        self.device = device
        self.shim = shim
        self.expect_query = expect_query
        self.rng = random.Random(seed)
        self.length = 0
        self.n_calls = 0
        self.info: dict[str, object] = {}
        self._process: subprocess.Popen[str] | None = None
        self._stderr: Path | None = None

    def __enter__(self) -> "MsaProposer":
        """Start the runner, load the model and hand it the alignment."""
        # stderr goes to a file, not a pipe: the runner prints a checkpoint-download progress bar on
        # first use, and an unread pipe of that size deadlocks the child.
        handle = tempfile.NamedTemporaryFile("w", suffix=".evodiff.log", delete=False)
        self._stderr = Path(handle.name)
        self._process = subprocess.Popen(
            [self.shim, "--serve", "--model", self.model, "--device", self.device],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=handle, text=True, bufsize=1,
        )
        self.info = dict(self._request({"op": "ping"}))
        loaded = self._request({
            "op": "msa", "path": str(self.a3m), "n_sequences": self.n_sequences,
            "max_seq_len": self.max_seq_len, "selection_type": self.selection_type, "seed": self.seed,
        })
        self.length = int(loaded["length"])
        self.info["n_sequences"] = loaded["n_sequences"]
        loaded_query = str(loaded.get("query", ""))
        if self.expect_query and loaded_query and loaded_query != self.expect_query:
            raise ValueError(
                f"{self.a3m} holds a different query than the one written for it: the runner read "
                f"{len(loaded_query)} columns starting {loaded_query[:16]!r}, this proposer was "
                f"built for {len(self.expect_query)} starting {self.expect_query[:16]!r}. Another "
                "run is writing the same alignment paths; give each its own --workdir."
            )
        return self

    def __exit__(self, kind: type[BaseException] | None, value: BaseException | None,
                 traceback: TracebackType | None) -> None:
        """Shut the runner down, killing it if it will not exit."""
        process = self._process
        self._process = None
        if process is None:
            return
        try:
            if process.poll() is None and process.stdin is not None:
                process.stdin.write(json.dumps({"op": "close"}) + "\n")
                process.stdin.flush()
            process.wait(timeout=30)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            process.kill()
            process.wait(timeout=30)

    def __call__(self, working: list[str | None], position: int, blocked: str) -> str:
        """Fill one masked alignment column of the query, conditioned on the whole MSA."""
        if self._process is None:
            raise ValueError("MsaProposer must be used as a context manager (`with MsaProposer(...)`)")
        response = self._request({
            "op": "fill", "query": row_from_working(working, self.length), "position": position,
        })
        self.n_calls += 1
        logits = {str(k): float(v) for k, v in response["logits"].items()}
        return softmax_choice(logits, blocked, self.temperature, self.rng)

    def generate_query(self, penalty_value: float = 2.0) -> str:
        """Run evodiff's own ``generate_query_oadm_msa_simple`` on this alignment, verbatim."""
        response = self._request({"op": "generate_query", "penalty_value": penalty_value})
        return str(response["query"]).replace(GAP, "")

    def _request(self, payload: dict[str, object]) -> dict[str, Any]:
        """Send one protocol line and read one response line."""
        process = self._process
        if process is None or process.stdin is None or process.stdout is None:
            raise ValueError("MsaProposer must be used as a context manager (`with MsaProposer(...)`)")
        try:
            process.stdin.write(json.dumps(payload) + "\n")
            process.stdin.flush()
            line = process.stdout.readline()
        except (BrokenPipeError, OSError) as error:
            raise RuntimeError(f"{self.shim} died during {payload.get('op')!r}{self._log()}") from error
        if not line:
            raise RuntimeError(f"{self.shim} produced no response to {payload.get('op')!r}{self._log()}")
        try:
            response = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"{self.shim} wrote non-JSON: {line.strip()[:400]!r}{self._log()}") from error
        if "error" in response:
            raise RuntimeError(f"{self.shim} failed on {payload.get('op')!r}: {response['error']}{self._log()}")
        return response

    def _log(self) -> str:
        """Return the runner's captured stderr tail for an error message, or ``""``."""
        if self._stderr is None or not self._stderr.exists():
            return ""
        tail = self._stderr.read_text().strip().splitlines()[-20:]
        return ("\n  " + "\n  ".join(tail)) if tail else ""
