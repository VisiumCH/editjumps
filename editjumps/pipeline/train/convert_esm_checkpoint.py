"""Fallback: fetch an ESM-2 checkpoint via `fair-esm` and convert to HF format. `pretrain_esm.py`."""

from pathlib import Path
from typing import Annotated

import typer

from editjumps.core.utils import get_logger

logger = get_logger(__file__)

# hidden_size / num_hidden_layers / num_attention_heads / intermediate_size per fair-esm loader name
MODEL_DIMS = {
    "esm2_t6_8M_UR50D": dict(hidden_size=320, num_hidden_layers=6, num_attention_heads=20, intermediate_size=1280),
    "esm2_t12_35M_UR50D": dict(hidden_size=480, num_hidden_layers=12, num_attention_heads=20, intermediate_size=1920),
    "esm2_t30_150M_UR50D": dict(hidden_size=640, num_hidden_layers=30, num_attention_heads=20, intermediate_size=2560),
    "esm2_t33_650M_UR50D": dict(hidden_size=1280, num_hidden_layers=33, num_attention_heads=20, intermediate_size=5120),
}


def convert(loader_name: str, output_folder: Path) -> Path:
    """Download an ESM-2 checkpoint via fair-esm and save it in HF format."""
    # Imported lazily so the `editjumps` CLI and `import convert_esm_checkpoint` stay usable without
    # the optional `train`/`analysis` deps (fair-esm/transformers).
    import esm
    from transformers import EsmConfig, EsmForMaskedLM, EsmTokenizer

    if loader_name not in MODEL_DIMS:
        raise ValueError(f"Unknown loader_name {loader_name!r}; supported: {sorted(MODEL_DIMS)}")

    logger.info(f"downloading {loader_name} via fair-esm (dl.fbaipublicfiles.com) ...")
    fair_model, alphabet = getattr(esm.pretrained, loader_name)()
    sd = fair_model.state_dict()
    dims = MODEL_DIMS[loader_name]
    n_layers = dims["num_hidden_layers"]

    config = EsmConfig(
        vocab_size=len(alphabet.tok_to_idx),
        pad_token_id=alphabet.padding_idx,
        mask_token_id=alphabet.mask_idx,
        max_position_embeddings=1026,
        layer_norm_eps=1e-5,
        position_embedding_type="rotary",
        emb_layer_norm_before=False,
        token_dropout=True,
        hidden_size=dims["hidden_size"],
        num_hidden_layers=dims["num_hidden_layers"],
        num_attention_heads=dims["num_attention_heads"],
        intermediate_size=dims["intermediate_size"],
    )
    hf_model = EsmForMaskedLM(config)

    new_sd = {
        "esm.embeddings.word_embeddings.weight": sd["embed_tokens.weight"],
        "esm.rotary_embeddings.inv_freq": sd["layers.0.self_attn.rot_emb.inv_freq"],
        "esm.encoder.emb_layer_norm_after.weight": sd["emb_layer_norm_after.weight"],
        "esm.encoder.emb_layer_norm_after.bias": sd["emb_layer_norm_after.bias"],
        "esm.contact_head.regression.weight": sd["contact_head.regression.weight"],
        "esm.contact_head.regression.bias": sd["contact_head.regression.bias"],
        "lm_head.bias": sd["lm_head.bias"],
        "lm_head.dense.weight": sd["lm_head.dense.weight"],
        "lm_head.dense.bias": sd["lm_head.dense.bias"],
        "lm_head.layer_norm.weight": sd["lm_head.layer_norm.weight"],
        "lm_head.layer_norm.bias": sd["lm_head.layer_norm.bias"],
        "lm_head.decoder.weight": sd["lm_head.weight"],
    }
    for i in range(n_layers):
        p, q = f"layers.{i}.", f"esm.encoder.layer.{i}."
        new_sd[q + "attention.self.query.weight"] = sd[p + "self_attn.q_proj.weight"]
        new_sd[q + "attention.self.query.bias"] = sd[p + "self_attn.q_proj.bias"]
        new_sd[q + "attention.self.key.weight"] = sd[p + "self_attn.k_proj.weight"]
        new_sd[q + "attention.self.key.bias"] = sd[p + "self_attn.k_proj.bias"]
        new_sd[q + "attention.self.value.weight"] = sd[p + "self_attn.v_proj.weight"]
        new_sd[q + "attention.self.value.bias"] = sd[p + "self_attn.v_proj.bias"]
        new_sd[q + "attention.output.dense.weight"] = sd[p + "self_attn.out_proj.weight"]
        new_sd[q + "attention.output.dense.bias"] = sd[p + "self_attn.out_proj.bias"]
        new_sd[q + "attention.LayerNorm.weight"] = sd[p + "self_attn_layer_norm.weight"]
        new_sd[q + "attention.LayerNorm.bias"] = sd[p + "self_attn_layer_norm.bias"]
        new_sd[q + "intermediate.dense.weight"] = sd[p + "fc1.weight"]
        new_sd[q + "intermediate.dense.bias"] = sd[p + "fc1.bias"]
        new_sd[q + "output.dense.weight"] = sd[p + "fc2.weight"]
        new_sd[q + "output.dense.bias"] = sd[p + "fc2.bias"]
        new_sd[q + "LayerNorm.weight"] = sd[p + "final_layer_norm.weight"]
        new_sd[q + "LayerNorm.bias"] = sd[p + "final_layer_norm.bias"]

    result = hf_model.load_state_dict(new_sd, strict=True)
    assert not result.missing_keys and not result.unexpected_keys, result

    output_folder.mkdir(parents=True, exist_ok=True)
    hf_model.save_pretrained(output_folder)

    idx_to_tok = {idx: tok for tok, idx in alphabet.tok_to_idx.items()}
    vocab = [idx_to_tok[i] for i in range(len(idx_to_tok))]
    (output_folder / "vocab.txt").write_text("\n".join(vocab))
    EsmTokenizer(vocab_file=str(output_folder / "vocab.txt")).save_pretrained(output_folder)

    logger.info(f"saved HF-format checkpoint to {output_folder}")
    return output_folder


def main(
    loader_name: Annotated[str, typer.Option(help="fair-esm loader function name")] = "esm2_t12_35M_UR50D",
    output_folder: Annotated[Path, typer.Option()] = Path("data/pretrain/base_checkpoints/esm2_t12_35M_UR50D"),
) -> None:
    """Convert a fair-esm checkpoint to HF format."""
    convert(loader_name, output_folder)


if __name__ == "__main__":
    typer.run(main)
