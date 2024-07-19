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

from itertools import chain

import numpy as np
import torch
import torch.nn as nn
from apex.contrib.transducer import TransducerJoint
from beartype import beartype
from beartype.typing import Optional
from jaxtyping import Int, jaxtyped

from caiman_asr_train.rnnt.rnn import rnn
from caiman_asr_train.rnnt.state import EncoderState, PredNetState, RNNTState
from caiman_asr_train.train_utils.distributed import print_once
from caiman_asr_train.train_utils.rsp import (
    get_pred_net_state,
    maybe_get_last_nonpadded,
)


class StackTime(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = int(factor)

    def forward(self, x, x_lens):
        # T, B, U
        seq = [x]
        for i in range(1, self.factor):
            tmp = torch.zeros_like(x)
            tmp[:-i, :, :] = x[i:, :, :]
            seq.append(tmp)
        # x_lens = torch.ceil(x_lens.float() / self.factor).int()
        x_lens = (x_lens.int() + self.factor - 1) // self.factor
        return torch.cat(seq, dim=2)[:: self.factor, :, :], x_lens


class RNNT(nn.Module):
    """A Recurrent Neural Network Transducer (RNN-T).

    Args:
        in_feats: Number of input features per step per batch.
        forget_gate_bias: Total initialized value of the bias used in the
            forget gate. Set to None to use PyTorch's default initialisation.
            (See: http://proceedings.mlr.press/v37/jozefowicz15.pdf)
        enc_batch_norm: Use batch normalization in encoder network if true.
        enc_n_hid: Internal hidden unit size of the encoder.
        enc_pre_rnn_layers: Encoder number of layers before Stack Time.
        enc_post_rnn_layers: Encoder number of layers after Stack Time.
        pred_n_hid:  Internal hidden unit size of the prediction network.
        pred_rnn_layers: Prediction network number of layers.
        joint_n_hid: Internal hidden unit size of the joint network.
        *lr_factor: network or layer-specific learning rate factor.
        joint_apex_transducer: If not None, use the TransducerJoint implementation.
            There are two valid non-None values of
            joint_apex_transducer={'pack', 'not_pack'}. 'pack' means that the output
            from the joint, and hence the RNNT model, will have padded elements removed.
            This is ignored at inference time as the torch version is used.
        joint_apex_relu_dropout: If True, this requires bool(joint_apex_transducer)=True,
            the joint network's relu and dropout calculations take place in the
            TransducerJoint implementation rather than native PyTorch. This is ignored
            at inference time as the torch version is used.
    """

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
        enc_rw_dropout=0.0,
        pred_rw_dropout=0.0,
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
    ):
        super(RNNT, self).__init__()
        if joint_apex_relu_dropout and not joint_apex_transducer:
            raise ValueError(
                "Can't have joint_apex_relu_dropout=True without "
                "bool(joint_apex_transducer)==True"
            )
        if joint_apex_transducer is not None:
            assert joint_apex_transducer in {"pack", "not_pack"}

        self._module_to_lr_factor = {
            "encoder": enc_lr_factor,
            "prediction": pred_lr_factor,
            "joint_enc": joint_enc_lr_factor,
            "joint_pred": joint_pred_lr_factor,
            "joint_net": joint_net_lr_factor,
        }

        self.pred_n_hid = pred_n_hid

        pre_rnn_input_size = in_feats

        self.enc_stack_time_factor = enc_stack_time_factor
        post_rnn_input_size = self.enc_stack_time_factor * enc_n_hid

        if custom_lstm:
            print_once("Using Custom LSTM")
            if quantize:
                print_once("Using Quantization")
            if enc_rw_dropout:
                print_once(f"Using encoder recurrent weight dropout {enc_rw_dropout}")
            if pred_rw_dropout:
                print_once(
                    f"Using prediction recurrent weight dropout {pred_rw_dropout}"
                )
        else:
            print_once("Using PyTorch LSTM")

        enc_mod = {}
        enc_mod["pre_rnn"] = rnn(
            input_size=pre_rnn_input_size,
            hidden_size=enc_n_hid,
            num_layers=enc_pre_rnn_layers,
            batch_norm=enc_batch_norm,
            forget_gate_bias=forget_gate_bias,
            custom_lstm=custom_lstm,
            quantize=quantize,
            rw_dropout=enc_rw_dropout,
            hidden_hidden_bias_scale=hidden_hidden_bias_scale,
            weights_init_scale=weights_init_scale,
            dropout=enc_dropout,
            tensor_name="pre_rnn",
            gpu_unavailable=gpu_unavailable,
        )

        enc_mod["stack_time"] = StackTime(self.enc_stack_time_factor)

        enc_mod["post_rnn"] = rnn(
            input_size=post_rnn_input_size,
            hidden_size=enc_n_hid,
            num_layers=enc_post_rnn_layers,
            batch_norm=enc_batch_norm,
            forget_gate_bias=forget_gate_bias,
            custom_lstm=custom_lstm,
            quantize=quantize,
            rw_dropout=enc_rw_dropout,
            hidden_hidden_bias_scale=hidden_hidden_bias_scale,
            weights_init_scale=weights_init_scale,
            dropout=enc_dropout,
            tensor_name="post_rnn",
            gpu_unavailable=gpu_unavailable,
        )

        self.encoder = torch.nn.ModuleDict(enc_mod)

        # Turn off encoder gradients if enc_freeze is True
        self.encoder.requires_grad_(not enc_freeze)

        pred_embed = torch.nn.Embedding(n_classes - 1, pred_n_hid)

        self.prediction = torch.nn.ModuleDict(
            {
                "embed": pred_embed,
                "dec_rnn": rnn(
                    input_size=pred_n_hid,
                    hidden_size=pred_n_hid,
                    num_layers=pred_rnn_layers,
                    batch_norm=pred_batch_norm,
                    forget_gate_bias=forget_gate_bias,
                    custom_lstm=custom_lstm,
                    quantize=quantize,
                    rw_dropout=pred_rw_dropout,
                    hidden_hidden_bias_scale=hidden_hidden_bias_scale,
                    weights_init_scale=weights_init_scale,
                    dropout=pred_dropout,
                    tensor_name="dec_rnn",
                    gpu_unavailable=gpu_unavailable,
                ),
            }
        )

        self.joint_pred = torch.nn.Linear(pred_n_hid, joint_n_hid)
        self.joint_enc = torch.nn.Linear(enc_n_hid, joint_n_hid)

        self.joint_net = nn.Sequential(
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=joint_dropout),
            torch.nn.Linear(joint_n_hid, n_classes),
        )

        # ReLU & Dropout will be applied in TransducerJoint during training if
        # joint_apex_transducer=joint_apex_relu_dropout=True. However, during
        # inference TransducerJoint is not used, but the torch version is used.
        self.relu_drop = self.joint_net[:2]
        self.joint_fc = self.joint_net[-1]

        self.joint_apex_transducer = joint_apex_transducer
        if joint_apex_transducer is not None:
            pack_output = self.joint_apex_transducer == "pack"
            if joint_apex_relu_dropout:
                self.apex_joint = TransducerJoint(
                    pack_output=pack_output,
                    relu=True,
                    dropout=True,
                    dropout_prob=joint_dropout,
                )
            else:
                self.apex_joint = TransducerJoint(pack_output=pack_output)

    def enc_pred(
        self,
        x,
        x_lens,
        y,
        y_lens,
        pred_net_state: Optional[PredNetState] = None,
        enc_state: Optional[EncoderState] = None,
    ):
        """
        Returns tuple of tuples of encoder and pred net outputs and their lengths.
        """
        return self.enc_pred_static(
            x,
            x_lens,
            y,
            y_lens,
            self.encode,
            self.predict,
            pred_net_state=pred_net_state,
            enc_state=enc_state,
        )

    @staticmethod
    def enc_pred_static(
        x,
        x_lens,
        y,
        y_lens,
        encode,
        predict,
        pred_net_state: Optional[PredNetState] = None,
        enc_state: Optional[EncoderState] = None,
    ):
        y = label_collate(y)

        f, x_lens, new_enc_state = encode(x, x_lens, enc_state=enc_state)

        g, _, all_pred_hid = predict(
            y,
            pred_state=pred_net_state.next_to_last_pred_state
            if pred_net_state
            else None,
            add_sos=True,
            special_sos=pred_net_state.last_token if pred_net_state else None,
        )
        # predict adds +1 to y_lens
        g_lens = y_lens + 1
        pred_net_state = get_pred_net_state(y, all_pred_hid, y_lens, g_lens)
        if new_enc_state is not None and pred_net_state is not None:
            rnnt_state = RNNTState(
                enc_state=new_enc_state, pred_net_state=pred_net_state
            )
        else:
            rnnt_state = None
        return (f, x_lens), (g, g_lens), rnnt_state

    def forward(
        self,
        x,
        x_lens,
        y,
        y_lens,
        pred_net_state: Optional[PredNetState] = None,
        batch_offset=None,
        enc_state: Optional[EncoderState] = None,
    ):
        (f, x_lens), (g, g_lens), new_rnnt_state = self.enc_pred(
            x, x_lens, y, y_lens, pred_net_state=pred_net_state, enc_state=enc_state
        )
        out = self.joint(f, g, x_lens, g_lens, batch_offset)

        return out, x_lens, new_rnnt_state

    def encode(self, x, x_lens, enc_state: Optional[EncoderState] = None):
        """
        Args:
            x: tuple of ``(input, input_lens)``. ``input`` has shape (T, B, I),
                ``input_lens`` has shape ``(B,)``.

        Returns:
            f: tuple of ``(output, output_lens)``. ``output`` has shape
                (B, T, H), ``output_lens``
        """
        x, _, all_pre_rnn_hid = self.encoder["pre_rnn"](
            x, enc_state.pre_rnn if enc_state else None
        )
        staggered_pre_rnn_hid = maybe_get_last_nonpadded(all_pre_rnn_hid, x_lens)
        x, x_lens = self.encoder["stack_time"](x, x_lens)
        x, _, all_post_rnn_hid = self.encoder["post_rnn"](
            x, enc_state.post_rnn if enc_state else None
        )
        staggered_post_rnn_hid = maybe_get_last_nonpadded(all_post_rnn_hid, x_lens)
        x = x.transpose(0, 1)
        x = self.joint_enc(x)
        if all_pre_rnn_hid is not None and all_post_rnn_hid is not None:
            new_enc_state = EncoderState(
                pre_rnn=staggered_pre_rnn_hid,
                post_rnn=staggered_post_rnn_hid,
            )
        else:
            new_enc_state = None
        return x, x_lens, new_enc_state

    @jaxtyped(typechecker=beartype)
    def predict(
        self,
        y,
        pred_state=None,
        add_sos: bool = True,
        special_sos: Optional[Int[torch.Tensor, "B 1"]] = None,
    ):
        """
        B - batch size
        U - label length
        H - Hidden dimension size
        L - Number of decoder layers = 2

        Args:
            y: (B, U)
            special_sos: If not None, use this as the "start of sequence" symbol and
                then embed it. If None, use a zero vector as the "start of sequence"
                embedding.

        Returns:
            Tuple (g, hid) where:
                g: (B, U + 1, H)
                hid: (h, c) where h is the final sequence hidden state and c is
                    the final cell state:
                        h (tensor), shape (L, B, H)
                        c (tensor), shape (L, B, H)
        """
        if y is not None:
            # (B, U) -> (B, U, H)
            y = self.prediction["embed"](y)
        else:
            B = 1 if pred_state is None else pred_state[0].size(1)
            y = torch.zeros((B, 1, self.pred_n_hid)).to(
                device=self.joint_enc.weight.device, dtype=self.joint_enc.weight.dtype
            )

        # prepend blank "start of sequence" symbol
        if add_sos:
            B, U, H = y.shape
            sos_embedding = (
                torch.zeros((B, 1, H))
                if special_sos is None
                else self.prediction["embed"](special_sos)
            )
            start = sos_embedding.to(device=y.device, dtype=y.dtype)
            y = torch.cat([start, y], dim=1).contiguous()  # (B, U + 1, H)
        else:
            start = None  # makes del call later easier

        y = y.transpose(0, 1)  # (U + 1, B, H)
        g, hid, all_hid = self.prediction["dec_rnn"](y, pred_state)
        g = g.transpose(0, 1)  # (B, U + 1, H)
        del y, start, pred_state
        g = self.joint_pred(g)
        return g, hid, all_hid

    def joint(self, f, g, f_len=None, g_len=None, batch_offset=None):
        """
        f should be shape (B, T, H)
        g should be shape (B, U + 1, H)

        returns:
            logits of shape (B, T, U + 1, K + 1) if not self.apex_joint.pack_output. If
            pack_output=True, all padded logits are removed from the output and the
            returned tensor is of the shape (<number non-padded logits>, K + 1). The
            number of these non-padded logits ('packed_batch' in call to
            TransducerJoint) is strictly <= (B * T * (U + 1)).
        """
        if self.joint_apex_transducer is None or f_len is None or g_len is None:
            h = self.relu_drop(self.torch_transducer_joint(f, g, f_len, g_len))
        else:
            assert batch_offset is not None
            h = self.apex_joint(
                f,
                g,
                f_len,
                g_len,
                batch_offset=batch_offset,
                packed_batch=batch_offset[-1].item(),
            )
            if not self.apex_joint.relu:
                h = self.relu_drop(h)

        res = self.joint_fc(h)

        del f, g, h
        return res

    @staticmethod
    def torch_transducer_joint(f, g, f_len=None, g_len=None):
        f = f.unsqueeze(dim=2)  # (B, T, 1, H)
        g = g.unsqueeze(dim=1)  # (B, 1, U + 1, H)
        h = f + g  # (B, T, U + 1, H)
        del f, g
        return h

    def param_groups(self, lr, return_module_name=False):
        out = []
        for name, lr_factor in self._module_to_lr_factor.items():
            res = {
                "params": self._chain_params(getattr(self, name)),
                "lr": lr * lr_factor,
            }
            if return_module_name:
                res["module_name"] = name
            out.append(res)
        return out

    def _chain_params(self, *layers):
        return chain(*[layer.parameters() for layer in layers])

    def state_dict(self):
        """
        Return model state dict with joint_fc.weight and joint_fc.bias keys removed,
        as they are exact duplicates of joint_net.2.weight and joint_net.2.bias,
        respectively.
        """
        sd = super().state_dict()
        sd.pop("joint_fc.weight")
        sd.pop("joint_fc.bias")
        return sd

    def load_state_dict(self, state_dict, strict=True):
        """
        Add joint_fc keys back to loaded state dict,
        by duplicating joint_net.2 parameters.
        """
        state_dict["joint_fc.weight"] = (
            state_dict["joint_net.2.weight"].detach().clone()
        )
        state_dict["joint_fc.bias"] = state_dict["joint_net.2.bias"].detach().clone()
        super().load_state_dict(state_dict, strict=strict)


def label_collate(labels):
    """Collates the label inputs for the rnn-t prediction network.

    If `labels` is already in torch.Tensor form this is a no-op.

    Args:
        labels: A torch.Tensor List of label indexes or a torch.Tensor.

    Returns:
        A padded torch.Tensor of shape (batch, max_seq_len).
    """

    if isinstance(labels, torch.Tensor):
        return labels.type(torch.int64)
    if not isinstance(labels, (list, tuple)):
        raise ValueError(f"`labels` should be a list or tensor not {type(labels)}")

    batch_size = len(labels)
    max_len = max(len(label) for label in labels)

    cat_labels = np.full((batch_size, max_len), fill_value=0.0, dtype=np.int32)
    for e, l in enumerate(labels):
        cat_labels[e, : len(l)] = l
    labels = torch.LongTensor(cat_labels)

    return labels
