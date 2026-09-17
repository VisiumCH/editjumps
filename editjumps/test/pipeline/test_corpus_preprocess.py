"""Building the pretraining corpus: a tokenizer-valid chain separator and the chain-handling axis."""

import inspect

import pytest
import yaml

from editjumps.pipeline.preprocess.pretrain import download_oas

# ESM-2's tokenizer vocabulary (all esm2_t*_UR50D sizes share one alphabet). A chain
# separator outside this set maps to <unk>, collapsing every VH/VL boundary to one token.
ESM2_VOCAB = {
    "<cls>", "<pad>", "<eos>", "<unk>",
    "L", "A", "G", "V", "S", "E", "R", "T", "I", "D", "P", "K", "Q", "N",
    "F", "Y", "M", "H", "W", "C", "X", "B", "U", "Z", "O",
    ".", "-", "<null_1>", "<mask>",
}


def test_download_oas_default_pair_sep_is_tokenizer_valid() -> None:
    """Any separator not in the ESM-2 vocab collapses to <unk>, so guard the default."""
    build_corpus_default = inspect.signature(download_oas.build_corpus).parameters["pair_sep"].default
    main_default = inspect.signature(download_oas.main).parameters["pair_sep"].default
    assert build_corpus_default in ESM2_VOCAB, f"build_corpus pair_sep {build_corpus_default!r} not in ESM-2 vocab"
    assert main_default in ESM2_VOCAB, f"main pair_sep default {main_default!r} not in ESM-2 vocab"


def test_params_yaml_pair_sep_is_tokenizer_valid() -> None:
    """Params.yaml's oas.pair_sep (what dvc repro actually uses) must be tokenizer-valid."""
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    pair_sep = params["oas"]["pair_sep"]
    assert pair_sep in ESM2_VOCAB, f"params.yaml oas.pair_sep {pair_sep!r} not in ESM-2 vocab"


def test_chain_handling_is_a_design_axis_and_separate_is_deferred() -> None:
    """`chain_handling` is a tracked design axis; only 'combine' is implemented, 'separate' defers."""
    from pathlib import Path

    from editjumps.core.utils import DESIGN_AXES
    from editjumps.pipeline.preprocess.pretrain.split_corpus import split_corpus

    assert "chain_handling" in DESIGN_AXES
    with pytest.raises(NotImplementedError):
        # raises before touching any file (the guard is the first statement)
        split_corpus(Path("_"), Path("_"), Path("_"), chain_handling="separate")
