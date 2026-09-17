from dataclasses import dataclass
from itertools import starmap
from typing import overload
from warnings import warn

import torch
import torch.nn as nn
from beartype import beartype
from beartype.typing import Optional, Tuple, Union
from torch import Tensor

from caiman_asr_train.core.batches import BatchFirst, TimeFirst
from caiman_asr_train.core.lstm import (
    LSTM,
    BackendConfig,
    ConfigLSTM,
    Hidden,
)
from caiman_asr_train.core.packing import packed_batch_offsets
from caiman_asr_train.core.pytree import register_dataclass_as_pytree


@beartype
@dataclass
class ConfigRNNT:
    # Number of tokens/output classes (including blank)
    n_classes: int

    # Encoder
    in_feats: int

    enc_n_hid: int
    enc_pre_rnn_layers: int
    enc_post_rnn_layers: int
    enc_stack_time_factor: int
    enc_dropout: float
    prefix_token_map: Optional[dict[int, int]]  # token_idx -> enc_embd_idx

    # Prediction
    pred_n_hid: int
    pred_rnn_layers: int
    pred_dropout: float

    # Shared between encoder and prediction
    backend: BackendConfig

    weights_init_scale: Optional[float]
    forget_gate_bias: Optional[float]
    hidden_hidden_bias_scale: Optional[float]

    # Joint
    joint_n_hid: int
    joint_dropout: float


@register_dataclass_as_pytree
@beartype
@dataclass
class StateEncoder:
    pre_stack: Hidden
    aft_stack: Hidden


@register_dataclass_as_pytree
@beartype
@dataclass
class StatePred:
    """
    Information for initial state to continue from.
    """

    state: Hidden
    embed: Tensor  # [T=1, B, H]


class RNNT(nn.Module):
    @beartype
    def __init__(self, config: ConfigRNNT):

        super().__init__()

        self.config = config

        kwargs = dict(
            backend=config.backend,
            weights_init_scale=config.weights_init_scale,
            forget_gate_bias=config.forget_gate_bias,
            hidden_hidden_bias_scale=config.hidden_hidden_bias_scale,
        )

        # === Encoder setup

        self.encoder = torch.nn.ModuleDict(
            dict(
                pre_rnn=LSTM(
                    ConfigLSTM(
                        input_size=config.in_feats,
                        hidden_size=config.enc_n_hid,
                        num_layers=config.enc_pre_rnn_layers,
                        dropout=config.enc_dropout,
                        **kwargs,  # type: ignore
                    )
                ),
                post_rnn=LSTM(
                    ConfigLSTM(
                        input_size=config.enc_stack_time_factor * config.enc_n_hid,
                        hidden_size=config.enc_n_hid,
                        num_layers=config.enc_post_rnn_layers,
                        dropout=config.enc_dropout,
                        **kwargs,  # type: ignore
                    )
                ),
            )
        )

        if (token_map := config.prefix_token_map) is not None:
            self.prefix_token_map = nn.Buffer(_build_lookup_table(token_map))
            self.encoder["prefix_embd"] = nn.Embedding(len(token_map), config.in_feats)

        # === Prediction setup

        self.prediction = torch.nn.ModuleDict(
            dict(
                # Subtract one for the blank symbol
                embed=torch.nn.Embedding(config.n_classes - 1, config.pred_n_hid),
                dec_rnn=LSTM(
                    ConfigLSTM(
                        input_size=config.pred_n_hid,
                        hidden_size=config.pred_n_hid,
                        num_layers=config.pred_rnn_layers,
                        dropout=config.pred_dropout,
                        **kwargs,  # type: ignore
                    )
                ),
            )
        )

        # === Joint setup

        self.joint_pred = torch.nn.Linear(config.pred_n_hid, config.joint_n_hid)
        self.joint_enc = torch.nn.Linear(config.enc_n_hid, config.joint_n_hid)

        self.joint_net = nn.Sequential(
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=config.joint_dropout),
            torch.nn.Linear(config.joint_n_hid, config.n_classes),
        )

    @overload
    def encode(
        self,
        *,
        state: Optional[StateEncoder],
        prefix: TimeFirst,
        feats: TimeFirst,
    ) -> tuple[TimeFirst, Optional[StateEncoder]]: ...

    @overload
    def encode(
        self,
        *,
        state: Optional[StateEncoder],
        prefix: TimeFirst,
        feats: None,
    ) -> Optional[StateEncoder]: ...

    @overload
    def encode(
        self,
        *,
        state: Optional[StateEncoder],
        prefix: None,
        feats: TimeFirst,
    ) -> tuple[TimeFirst, Optional[StateEncoder]]: ...

    @beartype
    def encode(
        self,
        *,
        state: Optional[StateEncoder],
        prefix: Optional[TimeFirst],
        feats: Optional[TimeFirst],
        no_joint: bool = False,
    ) -> Union[Optional[StateEncoder], tuple[TimeFirst, Optional[StateEncoder]]]:
        """
        Args:
            feats:  (T, B, I), (B)
            prefix: (L, B   ), (B)
            no_joint: If True, don't run the joint network, just return the
                      encoder output (used by SW).

        None is equivalent to T/L = 0.

        This method:
            Stacks prefix with X:       -> (T + nL, B, I)
            Does time-stacking encoder: -> (T/n + L, B, H)
            Strips prefix:              -> (T/n, B, H)
            Joint network:              -> (T/n, B, J)
        """

        if prefix is not None:
            assert self.config.prefix_token_map is not None

        match prefix, feats:
            case None, None:
                raise ValueError("Need either feats or prefix")
            case None, TimeFirst():
                input = feats
            case TimeFirst(), None:
                input = self._embed_enc_prefix(prefix)
            case TimeFirst(), TimeFirst():
                assert prefix.data.shape[1] == feats.data.shape[1], "Batch mismatch"

                a, a_lens = self._embed_enc_prefix(prefix).as_tuple()
                b, b_lens = feats.as_tuple()

                # For this concat to be valid all the prefixes must be the same length
                assert (a_lens == a_lens[0]).all(), "Non uniform prefix lengths!"
                assert a_lens[0] == a.shape[0], "Padded prefix"
                assert (
                    a_lens.shape == b_lens.shape
                ), f"Feats don't match embedding batch {a_lens.shape} vs {b_lens.shape}"

                input = TimeFirst(torch.cat([a, b], dim=0), a_lens + b_lens)

        # Do the rnn -> stack -> rnn
        x, h_pre = _apply_rnn(
            self.encoder["pre_rnn"],  # type: ignore
            input=input,
            state=None if state is None else state.pre_stack,
        )
        x = self._stack_time(x)
        x, h_aft = _apply_rnn(
            self.encoder["post_rnn"],  # type: ignore
            input=x,
            state=None if state is None else state.aft_stack,
        )

        # Do the joint embedding
        if not no_joint:
            x = TimeFirst(self.joint_enc(x.data), x.lens)

        assert (h_aft is None) == (h_pre is None)

        if h_aft is not None and h_pre is not None:
            state = StateEncoder(pre_stack=h_pre, aft_stack=h_aft)
        else:
            state = None

        match prefix, feats:
            case None, None:
                assert False, "Unreachable due to check at start of method"
            case None, TimeFirst():
                # Nothing to strip
                return x, state
            case TimeFirst(), None:
                # No values to return
                return state
            case TimeFirst(), TimeFirst():
                # We need to strip the prefix from x

                # _embed_enc_prefix increased the length which was then reverted by
                # _stack_time so we can use the original prefix lens here.
                n = prefix.lens[0]
                return TimeFirst(x.data[n:], x.lens - n), state

    @beartype
    def _stack_time(self, input: TimeFirst) -> TimeFirst:
        x, x_lens = input.as_tuple()
        T, B, U = x.shape
        f = self.config.enc_stack_time_factor
        assert T % f == 0, f"Expected time dimension {T} to be divisible by {f}"
        assert (x_lens % f == 0).all(), f"Odd lengths {x_lens} for stack {f}"
        x = x.reshape(T // f, f, B, U)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(T // f, B, U * f)
        x_lens = (x_lens + f - 1) // f
        return TimeFirst(x, x_lens)

    @beartype
    def _embed_enc_prefix(self, prefix: TimeFirst) -> TimeFirst:

        n = self.config.enc_stack_time_factor

        pfx_idx = self.prefix_token_map[prefix.data]  # (T, B)
        pfx_emb = self.encoder["prefix_embd"](pfx_idx)  # (T, B, I)
        pfx_emb = pfx_emb.repeat_interleave(n, dim=0)  # (T*n, B, I)

        return TimeFirst(pfx_emb, n * prefix.lens)

    @beartype
    def _zeros(self, shape: tuple[int, ...]) -> Tensor:
        return torch.zeros(
            shape, device=self.joint_enc.weight.device, dtype=torch.float32
        )

    @beartype
    def sos_for(self, input: int | BatchFirst | TimeFirst) -> StatePred:
        """
        Generate a null SoS state, shaped to be compatible with "input".
        """

        B = _batch_size(input)

        L = self.config.pred_rnn_layers
        H = self.config.pred_n_hid

        return StatePred(
            Hidden(hidd=self._zeros((L, B, H)), cell=self._zeros((L, B, H))),
            embed=self._zeros((1, B, H)),
        )

    @beartype
    def default_enc_state_for(
        self, input: int | BatchFirst | TimeFirst
    ) -> StateEncoder:
        """
        Generate a zeroed default encoder state, shaped to be compatible with
        "input". The encoder counterpart of `sos_for`.

        Passing this to `encode(state=...)` is equivalent to passing
        `state=None`; it exists so that callers threading state across chunks
        have an explicit, correctly shaped value to start from.
        """

        B = _batch_size(input)

        H = self.config.enc_n_hid

        def hidden(layers: int) -> Hidden:
            return Hidden(
                hidd=self._zeros((layers, B, H)),
                cell=self._zeros((layers, B, H)),
            )

        return StateEncoder(
            pre_stack=hidden(self.config.enc_pre_rnn_layers),
            aft_stack=hidden(self.config.enc_post_rnn_layers),
        )

    @beartype
    def predict(
        self,
        *,
        state: StatePred,
        prefix: Optional[TimeFirst],
        tokens: Optional[TimeFirst],
    ) -> tuple[TimeFirst, Optional[StatePred]]:
        """
        Args:
            tokens: (U, B), (B)
            prefix: (L, B), (B)

        None is equivalent to T/L = 0.

        This method:
            Embeds prefix/tokens:      -> (U, B, H)/(L, B, H)
            Stacks prefix with tokens: -> (U + L, B, H)
            Adds state/SOS:            -> (U + L + 1, B, H)
            RNN:                       -> (U + L + 1, B, H)
            Strips prefix:             -> (U + 1, B, H)
            Joint network:             -> (U + 1, B, J)
        """

        B = state.state.hidd.shape[1]
        device = self.joint_enc.weight.device

        embd = [state.embed]
        embd_lens = torch.full((B,), 1, dtype=torch.int32, device=device)

        match prefix, tokens:
            case None, None:
                pass
            case None, TimeFirst():
                embd_lens += tokens.lens
                embd.append(self.prediction["embed"](tokens.data))
            case TimeFirst(), None:
                embd_lens += prefix.lens
                embd.append(self.prediction["embed"](prefix.data))
            case TimeFirst(), TimeFirst():
                a, a_lens = prefix.as_tuple()
                b, b_lens = tokens.as_tuple()

                # For this concat to be valid all the prefixes must be the same length
                assert (a_lens == a_lens[0]).all(), "Non uniform prefix lengths!"
                assert a_lens[0] == a.shape[0], "Padded prefix"
                assert a_lens.shape == b_lens.shape, f"{a_lens.shape} vs {b_lens.shape}"

                embd_lens += a_lens + b_lens
                embd.append(self.prediction["embed"](a))
                embd.append(self.prediction["embed"](b))

        input = TimeFirst(torch.cat(embd, dim=0), embd_lens)

        # Run the actual RNN.
        y, h = _apply_rnn(
            self.prediction["dec_rnn"],  # type: ignore
            input=input,
            state=state.state,
            penultimate=True,
        )

        if prefix is None and tokens is None:
            # No input -> return the SoS hidden state
            out_state = state
        elif h is not None:
            # Some input, we need the penultimate hidden state, need to
            # do some indexing to get the last embedding.

            # The preceding pick() inside _apply_rnn will be wrong unless:
            assert torch.all(embd_lens > 1), "Zero length input"

            # state out is penultimate hidden state + the last embedding:
            last = input.data[input.lens - 1, range(B)]
            # Unsqeeze to add time-dimension back, detach to avoid double backprop
            out_state = StatePred(state=h, embed=last[None].detach())
        else:
            # Some input, we want to return a hidden state but the LSTM didn't
            # give us one (i.e. TorchLSTM) so we return none.
            out_state = None

        if prefix is not None:
            # We need to drop prefix time-steps from the output
            n = prefix.lens[0]
            assert (n == prefix.lens).all(), "Non uniform drop lengths!"
            y.data = y.data[n:]
            y.lens = y.lens - n

        # Do the joint embedding
        y.data = self.joint_pred(y.data)

        return y, out_state

    @staticmethod
    def torch_transducer_joint(f: Tensor, g: Tensor) -> Tensor:

        f = f.unsqueeze(dim=2)  # (B, T, 1, J)
        g = g.unsqueeze(dim=1)  # (B, 1, U, J)

        return f + g  # (B, T, U, J)

    def pack_joint(self, f: BatchFirst, g: BatchFirst) -> Tuple[Tensor, Tensor]:
        """
        Maps inputs f, j to packed rnnt lattice logits returns
            packed_logits: (sum(t*u), K)
            batch_offsets: (B) cumulative sum of t*u for each batch member
        """
        f_, f_lens = f.as_tuple()
        g_, g_lens = g.as_tuple()

        h = self.torch_transducer_joint(f_, g_)  # (B, T, U, J)

        def pack(x, t, u):
            return x[:t, :u, :].flatten(0, 1)

        # [[T, U, K]...] -> [[u*t, K]...] -> [sum(t*u), K]
        h = torch.cat(list(starmap(pack, zip(h, f_lens, g_lens))))

        return self.joint_net(h), packed_batch_offsets(f_lens, g_lens)

    def joint(
        self,
        f: BatchFirst | Tensor,
        g: BatchFirst | Tensor,
        *,
        apply_joint_enc: bool = False,
    ) -> Tensor:
        """
        Maps inputs f, j to rnnt lattice logits -> (B, T, U, K)

        Args:
          apply_joint_enc: If True, apply the joint encoder to f before the
                           joint network (used by SW).
        """

        def pluck(x: BatchFirst | Tensor) -> Tensor:
            if isinstance(x, Tensor):
                return x
            else:
                return x.data

        f_ = pluck(f)
        g_ = pluck(g)

        if apply_joint_enc:
            f_ = self.joint_enc(f_)

        h = self.torch_transducer_joint(f_, g_)  # (B, T, U, J)

        return self.joint_net(h)  # (B, T, U, K)


@beartype
def _apply_rnn(
    rnn: LSTM,
    *,
    input: TimeFirst,
    state: Optional[Hidden],
    penultimate: bool = False,
) -> tuple[TimeFirst, Optional[Hidden]]:
    """
    Apply an RNN to input x with the given state and lens. If possible
    return the final hidden state for each batch member.
    """

    x, x_lens = input.as_tuple()

    assert x.shape[0] == x_lens.max(), f"{x.shape}, {x_lens.max()}"

    x, h, all = rnn(x, state)

    if all is not None:
        # The last hidden state for each member of the batch has to be
        # picked. This is not the last state in the tensor, since there's
        # padding. Hence index using lens to get the last non-padded state.
        @beartype
        def pick(y: Tensor) -> Tensor:
            return y[:, x_lens - (2 if penultimate else 1), range(len(x_lens))]

        # Detach the threaded state: it is only fed as the next step's initial
        # state (RSP), never backpropagated. Keeping it grad-connected would also
        # leave grad-requiring tensors in the DDP forward output.
        return TimeFirst(x, x_lens), Hidden(pick(all.hidd), pick(all.cell)).detach()

    if (x_lens == x_lens[0]).all() and not penultimate:
        # If they're all the same, ok to just return the last one.
        return TimeFirst(x, x_lens), h.detach()

    return TimeFirst(x, x_lens), None


@beartype
def _batch_size(input: int | BatchFirst | TimeFirst) -> int:
    """
    The batch size of "input", which may also be given directly as an int.
    """

    match input:
        case int():
            return input
        case BatchFirst(data):
            return data.shape[0]
        case TimeFirst(data):
            return data.shape[1]


@beartype
def _build_lookup_table(hashmap: dict[int, int]) -> Tensor:
    """
    Builds a vectorized lookup table from a dictionary.
    It validates all keys are small positive integers.
    """
    max_id = max(hashmap.keys())
    if max_id >= 100:
        warn(
            f"Unexpectedly large max_id ({max_id}). Prefix token_ids should be "
            "small integers (expected < 100). This may cause high memory usage.",
            UserWarning,
            stacklevel=2,
        )
    invalid_sentinel = torch.iinfo(torch.long).max
    lookup_tensor = torch.full((max_id + 1,), invalid_sentinel, dtype=torch.long)

    for k, v in hashmap.items():
        lookup_tensor[k] = v

    return lookup_tensor
