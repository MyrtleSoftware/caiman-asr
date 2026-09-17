# Copyright (c) 2019, Myrtle Software Limited. All rights reserved.
# Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from dataclasses import dataclass

import torch
from apex.contrib.transducer import TransducerJoint
from beartype import beartype
from beartype.typing import Optional

from caiman_asr_train.autotune.support import tilelang_supported
from caiman_asr_train.core.batches import BatchFirst, TimeFirst
from caiman_asr_train.core.lstm import (
    CustomConfig,
    LegacyConfig,
    TileConfig,
    TorchConfig,
)
from caiman_asr_train.core.model import RNNT as BaseRNNT
from caiman_asr_train.core.model import ConfigRNNT, StateEncoder, StatePred
from caiman_asr_train.core.packing import packed_batch_offsets
from caiman_asr_train.tilelstm.autotune.configs import tile_hidden_feasible


@beartype
@dataclass
class StateRNNT:
    """
    Package of encoder and prediction state.
    """

    enc: StateEncoder
    pred: StatePred


@beartype
@dataclass
class ForwardRNNT:
    """
    Return from the forward method
    """

    f: BatchFirst
    g: BatchFirst
    packed_logits: torch.Tensor
    batch_offset: torch.Tensor
    state: Optional[StateRNNT]


class RNNT(BaseRNNT):
    @beartype
    def __init__(
        self,
        n_classes,
        in_feats,
        enc_n_hid,
        enc_batch_norm,
        pred_batch_norm,
        enc_pre_rnn_layers,
        enc_post_rnn_layers,
        enc_stack_time_factor,
        enc_dropout,
        pred_dropout,
        joint_dropout,
        pred_n_hid,
        pred_rnn_layers,
        joint_n_hid,
        forget_gate_bias,
        custom_lstm=False,
        quantize=False,
        enc_rw_dropout=None,
        pred_rw_dropout=None,
        hidden_hidden_bias_scale=0.0,
        weights_init_scale=1.0,
        enc_lr_factor=1.0,
        pred_lr_factor=1.0,
        joint_enc_lr_factor=1.0,
        joint_pred_lr_factor=1.0,
        joint_net_lr_factor=1.0,
        joint_apex_transducer=None,
        joint_apex_relu_dropout=False,
        enc_freeze=False,
        gpu_unavailable=False,
        conditional_encoder=False,
        prefix_token_map: Optional[dict] = None,
    ):
        assert conditional_encoder == (
            prefix_token_map is not None
        ), "conditional_encoder and prefix_token_map must be set together"

        assert (
            enc_rw_dropout == pred_rw_dropout
        ), "dropout must be the same for encoder and predictor"

        assert not enc_batch_norm, "batch norm not supported for encoder"
        assert not pred_batch_norm, "batch norm not supported for predictor"

        assert joint_apex_transducer == "pack"

        if custom_lstm:
            if quantize or gpu_unavailable:
                backend = LegacyConfig(quantize=quantize, rw_dropout=enc_rw_dropout)
            elif (
                tilelang_supported()
                and (enc_rw_dropout is None or enc_rw_dropout == 0.0)
                and tile_hidden_feasible(enc_n_hid)
                and tile_hidden_feasible(pred_n_hid)
            ):
                # No recurrent-weight dropout on a new-enough GPU/torch, and both hidden
                # sizes are tileable by the kernels -> the faster TileLang LSTM, a
                # checkpoint-compatible drop-in for the custom CUDA LSTM. A hidden size
                # the kernels cannot tile (e.g. not a multiple of the WMMA k-atom) falls
                # through to the custom CUDA LSTM rather than hard-failing in autotune
                # selection.
                backend = TileConfig()
            else:
                backend = CustomConfig(rw_dropout=enc_rw_dropout)
        else:
            assert not quantize, "invalid arguments"
            assert enc_rw_dropout is None, "invalid arguments"
            assert pred_rw_dropout is None, "invalid arguments"

            backend = TorchConfig(return_all=False)

        config = ConfigRNNT(
            n_classes=n_classes,
            in_feats=in_feats,
            enc_n_hid=enc_n_hid,
            enc_pre_rnn_layers=enc_pre_rnn_layers,
            enc_post_rnn_layers=enc_post_rnn_layers,
            enc_stack_time_factor=enc_stack_time_factor,
            enc_dropout=enc_dropout,
            prefix_token_map=prefix_token_map,
            pred_n_hid=pred_n_hid,
            pred_rnn_layers=pred_rnn_layers,
            pred_dropout=pred_dropout,
            backend=backend,
            weights_init_scale=weights_init_scale,
            forget_gate_bias=forget_gate_bias,
            hidden_hidden_bias_scale=hidden_hidden_bias_scale,
            joint_n_hid=joint_n_hid,
            joint_dropout=joint_dropout,
        )

        self.joint_apex_relu_dropout = joint_apex_relu_dropout

        super().__init__(config=config)

        self._module_to_lr_factor = {
            "encoder": enc_lr_factor,
            "prediction": pred_lr_factor,
            "joint_enc": joint_enc_lr_factor,
            "joint_pred": joint_pred_lr_factor,
            "joint_net": joint_net_lr_factor,
        }

        self.apex_joint = (
            None
            if gpu_unavailable
            else TransducerJoint(
                pack_output=True,
                relu=joint_apex_relu_dropout,
                dropout=joint_apex_relu_dropout,
                dropout_prob=joint_dropout,
            )
        )

        # Turn off encoder gradients if enc_freeze is True
        self.encoder.requires_grad_(not enc_freeze)

    @beartype
    def param_groups(self, lr: float) -> list[dict]:
        return [
            dict(
                params=list(getattr(self, name).parameters()),
                lr=lr * factor,
            )
            for name, factor in self._module_to_lr_factor.items()
        ]

    @beartype
    def forward(
        self,
        *,
        feats: TimeFirst,
        tokens: TimeFirst,
        prefix: Optional[TimeFirst],
        pred_state: StatePred,
        enc_state: Optional[StateEncoder],
    ) -> ForwardRNNT:

        enc_prefix = prefix if self.config.prefix_token_map is not None else None

        f, e_state_out = self.encode(state=enc_state, prefix=enc_prefix, feats=feats)
        g, p_state_out = self.predict(state=pred_state, prefix=prefix, tokens=tokens)

        f_b = f.T.contiguous()
        g_b = g.T.contiguous()

        logits, batch_offset = self.joint(f_b, g_b)

        if p_state_out is not None and e_state_out is not None:
            state = StateRNNT(enc=e_state_out, pred=p_state_out)
        else:
            state = None

        return ForwardRNNT(
            f=f_b,
            g=g_b,
            packed_logits=logits,
            batch_offset=batch_offset,
            state=state,
        )

    def joint(
        self,
        f: BatchFirst,
        g: BatchFirst,
        packed_cells: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Joint network that uses apex transducer joint

        Returns:
            joint output [sum(u*t), K]
            batch offsets for packed sequences
        """

        if self.apex_joint is None:
            return super().pack_joint(f, g)

        f_, f_lens = f.as_tuple()
        g_, g_lens = g.as_tuple()

        batch_offset = packed_batch_offsets(f_lens, g_lens)

        # According to:
        #
        #   https://github.com/NVIDIA/apex/blob/5daec2ac39c66fe61f0c14de8eb20576a6d849d6/apex/contrib/transducer/transducer.py#L51
        #
        # the apex transducer joint expects `g_lens - 1` however, my reading of
        # their source code is that this is incorrect and historically we have
        # always used the length that includes the SoS here.

        h = self.apex_joint(
            f_,
            g_,
            f_lens,
            g_lens,
            batch_offset=batch_offset,
            packed_batch=(
                batch_offset[-1].item() if packed_cells is None else packed_cells
            ),
        )

        if self.joint_apex_relu_dropout:
            # Just projection, RelU/Drop are applied inside the Apex
            out = self.joint_net[-1](h)
        else:
            out = self.joint_net(h)

        # This is a ~ [B, T, U, J] tensor, clean up as early as possible
        del h

        return out, batch_offset
