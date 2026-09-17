"""Serve EvoDiff-MSA forward passes, inside evodiff's isolated environment."""

import argparse
import json
import sys

#: Checkpoint name -> its ``evodiff.pretrained`` loader.
LOADERS = {
    "msa-oa-dm-maxsub": "MSA_OA_DM_MAXSUB",
    "msa-oa-dm-randsub": "MSA_OA_DM_RANDSUB",
}

#: Residues a ``fill`` response reports a logit for.
PROPOSABLE = "ACDEFGHIKLMNPQRSTVWY"


def pick_device(requested: str) -> str:
    """Resolve ``requested`` (``auto``, or a torch device string) to a device. ``auto`` prefers CUDA."""
    import torch

    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


class Server:
    """One loaded EvoDiff-MSA checkpoint plus the alignment currently being decoded."""

    def __init__(self, model_name: str, device: str) -> None:
        """Load the `LOADERS` checkpoint ``model_name`` (downloading it once) onto ``device``."""
        import evodiff.pretrained

        self.model_name = model_name
        self.device = device
        loader = getattr(evodiff.pretrained, LOADERS[model_name])
        self.model, _collater, self.tokenizer, self.scheme = loader()
        self.model = self.model.to(device).eval()
        self.msa = None
        self._path = ""
        self.length = 0
        self.n_sequences = 0
        self.candidate_ids = [int(self.tokenizer.tokenizeMSA(residue)[0]) for residue in PROPOSABLE]

    def load_msa(self, path: str, n_sequences: int, max_seq_len: int, selection_type: str,
                 seed: int) -> dict:
        """Subsample one alignment with evodiff's own selector and tokenize it. ``path`` is an a3m with the."""
        import numpy as np
        import torch
        from evodiff.data import subsample_msa

        np.random.seed(seed)
        torch.manual_seed(seed)
        rows, query = subsample_msa(path, n_sequences=n_sequences, max_seq_len=max_seq_len,
                                    selection_type=selection_type)
        tokenized = np.array([self.tokenizer.tokenizeMSA(row).tolist() for row in rows])
        self.msa = torch.tensor(tokenized, dtype=torch.long, device=self.device).unsqueeze(0)
        self.n_sequences, self.length = self.msa.shape[1], self.msa.shape[2]
        return {"n_sequences": self.n_sequences, "length": self.length,
                "query_length": len(query), "query": query}

    def fill(self, query: str, position: int) -> dict:
        """One forward pass; return the logit of each standard amino acid at one query position."""
        import torch

        if self.msa is None:
            raise ValueError("no MSA loaded; send an `msa` request first")
        if len(query) != self.length:
            raise ValueError(f"query is {len(query)} characters but the alignment is {self.length} columns")
        if not 0 <= position < self.length:
            raise ValueError(f"position {position} outside 0..{self.length - 1}")
        sample = self.msa.clone()
        sample[0, 0, :] = torch.tensor(self.tokenizer.tokenizeMSA(query).tolist(),
                                       dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits = self.model(sample)[0, 0, position, :]
        return {"logits": {residue: float(logits[token]) for residue, token
                           in zip(PROPOSABLE, self.candidate_ids)}}

    def generate_query(self, penalty_value: float) -> dict:
        """Call evodiff's ``generate_query_oadm_msa_simple`` unchanged."""
        from evodiff.generate_msa import generate_query_oadm_msa_simple

        if self.msa is None:
            raise ValueError("no MSA loaded; send an `msa` request first")
        _sample, untokenized = generate_query_oadm_msa_simple(
            self._path, self.model, self.tokenizer, self.n_sequences, self.length,
            batch_size=1, penalty_value=penalty_value, device=self.device,
        )
        return {"query": untokenized[0][0]}

    def handle(self, request: dict) -> dict:
        """Dispatch one decoded protocol request to its handler and return the response body."""
        op = request.get("op")
        if op == "ping":
            return {"model": self.model_name, "device": self.device, "scheme": self.scheme}
        if op == "msa":
            self._path = request["path"]
            return self.load_msa(request["path"], int(request["n_sequences"]),
                                 int(request["max_seq_len"]), str(request["selection_type"]),
                                 int(request.get("seed", 0)))
        if op == "fill":
            return self.fill(str(request["query"]), int(request["position"]))
        if op == "generate_query":
            return self.generate_query(float(request.get("penalty_value", 2.0)))
        raise ValueError(f"unknown op {op!r}")


def serve(server: Server) -> int:
    """Read protocol lines from stdin until ``close`` or EOF, answering each with one line."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as error:
            print(json.dumps({"error": f"bad request line: {error}"}), flush=True)
            continue
        op = request.get("op")
        if op == "close":
            return 0
        try:
            response = server.handle(request)
        except Exception as error:  # noqa: BLE001 - any failure must reach the client as one line
            response = {"error": f"{type(error).__name__}: {error}"}
        print(json.dumps(response), flush=True)
    return 0


def main() -> int:
    """Parse arguments, load the checkpoint and serve (or self-test)."""
    parser = argparse.ArgumentParser(description="Serve EvoDiff-MSA forward passes over stdin/stdout.")
    parser.add_argument("--serve", action="store_true", help="read JSON protocol lines from stdin")
    parser.add_argument("--self-test", action="store_true",
                        help="load the checkpoint, build a toy MSA and do one forward pass")
    parser.add_argument("--model", default="msa-oa-dm-maxsub", choices=sorted(LOADERS))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = pick_device(args.device)
    server = Server(args.model, device)
    if args.self_test:
        return self_test(server)
    if not args.serve:
        parser.error("one of --serve / --self-test is required")
    return serve(server)


def self_test(server: Server) -> int:
    """Prove the checkpoint loads and a forward pass runs, without any of this repo present."""
    import tempfile
    from pathlib import Path

    query = "QVQLQESGGGLVQAGGSLRLSCAASGRTFSEYAMGWFRQAPGKEREFVATISWSGDSTYYADSVKG"
    rows = [query] + [query[:i] + "A" + query[i + 1:] for i in range(1, 9)]
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "probe.a3m"
        path.write_text("".join(f">row{i}\n{row}\n" for i, row in enumerate(rows)))
        shape = server.load_msa(str(path), len(rows), len(query), "MaxHamming", 0)
        masked = query[:5] + "#" + query[6:]
        logits = server.fill(masked, 5)["logits"]
    best = max(logits, key=lambda residue: logits[residue])
    print(f"ok: {server.model_name} on {server.device}, MSA {shape}, "
          f"argmax at position 5 = {best} (true residue {query[5]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
