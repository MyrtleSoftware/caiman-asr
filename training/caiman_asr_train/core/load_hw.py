from dataclasses import dataclass
from pathlib import Path

import torch
from beartype import beartype
from beartype.typing import Optional, Union

from caiman_asr_train.core.export_schema import (
    InputValSchema,
)
from caiman_asr_train.core.export_schema import (
    RNNTInferenceConfigSchema as InferenceConfig,
)
from caiman_asr_train.core.lstm import TorchConfig
from caiman_asr_train.core.model import RNNT, ConfigRNNT


@beartype
@dataclass
class Ngram:
    # Older checkpoints store this as raw bytes; newer ones as a tensor so it
    # can be `torch.load(..., mmap=True)`-mapped below instead of read fully
    # into RAM -- it can be many GB for a multilingual LM.
    binary: Union[bytes, torch.Tensor]
    scale: float


@beartype
@dataclass
class MelStats:
    means: torch.Tensor
    vars: torch.Tensor
    alpha: float


@beartype
@dataclass
class Checkpoint:
    model: RNNT
    data_config: InputValSchema
    sentpiece_model_binary: bytes
    ngram: Optional[Ngram]
    mel_stats: MelStats
    # The mau model compiler injects its compiled program into a hardware
    # checkpoint (see docs/src/inference/compiling_weights.md): None for a
    # plain hardware checkpoint, a dict for a CAIMAN-ASR one. Its shape is the
    # compiler's business, so it stays an unparameterised `dict`.
    mau_program: Optional[dict]
    version: str


@beartype
def load_hw_checkpoint(path: Path) -> Checkpoint:
    """
    Load a hardware checkpoint from disk into a TorchLSTM backed RNNT model.

    This function is primarily intended for SW's use.
    """
    # mmap=True lets large tensor-backed fields (e.g. `Ngram`) map straight
    # from disk instead of being read fully into RAM.
    return _load_hw_dict(
        torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    )


@beartype
def _load_hw_dict(ckpt: dict) -> Checkpoint:
    """
    Convert a hw checkpoint dict to a checkpoint.
    """
    inference_config = InferenceConfig(**ckpt["rnnt_config"])

    model_config = _model_schema_to_config(
        # We need to peak at the state_dict to get the vocab size
        n_classes=ckpt["state_dict"]["joint_net.2.weight"].shape[0],
        prefix_token_map=_prefix_token_map(ckpt),
        conf=inference_config,
    )

    ckpt_ngram = ckpt.get("ngram", {})
    n_binary = ckpt_ngram.get("binary", None)
    n_scale = ckpt_ngram.get("scale_factor", None)

    if n_binary is not None and n_scale is not None:
        ngram = Ngram(binary=n_binary, scale=n_scale)
    else:
        ngram = None

    model = RNNT(model_config)
    model.load_state_dict(ckpt["state_dict"])

    return Checkpoint(
        model=model,
        data_config=inference_config.input_val,
        sentpiece_model_binary=ckpt["sentpiece_model"],
        ngram=ngram,
        mel_stats=MelStats(
            means=ckpt["melmeans"],
            vars=ckpt["melvars"],
            alpha=ckpt["melalpha"],
        ),
        mau_program=ckpt.get("mau_program"),
        version=ckpt["version"],
    )


@beartype
def _prefix_token_map(ckpt: dict) -> Optional[dict[int, int]]:
    """
    Recover the encoder's prefix token map (token_idx -> enc_embd_idx), or None
    for a non-conditional encoder.

    Checkpoints from two eras have no usable dict and need the state dict's
    `prefix_token_map` lookup buffer inverted instead. That buffer is the same
    mapping in tensor form and every conditional-encoder model has written it,
    so it agrees with the dict by-construction.
    """

    state_dict = ckpt["state_dict"]

    if (raw := ckpt.get("raw_prefix_token_map", None)) is not None:
        return {int(token): int(index) for token, index in raw.items()}

    if (buffer := state_dict.get("prefix_token_map", None)) is None:
        # Not a conditional encoder: no prefix embedding to map into.
        return None

    assert "encoder.prefix_embd.weight" in state_dict, (
        "Checkpoint has a prefix_token_map buffer but no "
        "encoder.prefix_embd.weight to index into."
    )

    n_embd = state_dict["encoder.prefix_embd.weight"].shape[0]

    # _build_lookup_table fills unmapped token ids with a large sentinel, so a
    # valid embedding row index is what marks an entry as really mapped.
    token_map = {
        int(token): int(index)
        for token, index in enumerate(buffer.tolist())
        if 0 <= index < n_embd
    }

    assert sorted(token_map.values()) == list(range(n_embd)), (
        f"Prefix token map {token_map} recovered from the state dict does not "
        f"cover the {n_embd} rows of encoder.prefix_embd.weight exactly once."
    )

    return token_map


@beartype
def _model_schema_to_config(
    n_classes: int, prefix_token_map: Optional[dict[int, int]], conf: InferenceConfig
) -> ConfigRNNT:
    """
    Convert an inference config to a model config.
    """

    rnnt = conf.rnnt

    n_filt = conf.input_val.filterbank_features.n_filt
    factor = conf.input_val.frame_splicing.frame_stacking

    return ConfigRNNT(
        n_classes=n_classes,
        in_feats=n_filt * factor,
        enc_n_hid=rnnt.enc_n_hid,
        enc_pre_rnn_layers=rnnt.enc_pre_rnn_layers,
        enc_post_rnn_layers=rnnt.enc_post_rnn_layers,
        enc_stack_time_factor=rnnt.enc_stack_time_factor,
        enc_dropout=0.0,
        prefix_token_map=prefix_token_map,
        pred_n_hid=rnnt.pred_n_hid,
        pred_rnn_layers=rnnt.pred_rnn_layers,
        pred_dropout=0.0,
        backend=TorchConfig(return_all=True),
        weights_init_scale=None,
        forget_gate_bias=None,
        hidden_hidden_bias_scale=None,
        joint_n_hid=rnnt.joint_n_hid,
        joint_dropout=0.0,
    )
