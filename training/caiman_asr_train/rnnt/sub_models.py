from collections import Counter
from dataclasses import dataclass

import torch
from beartype import beartype
from beartype.typing import Any, Iterator, Mapping, Tuple, Union
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parameter import Parameter

from caiman_asr_train.rnnt.joint_loss import JointLoss, build_joint_loss
from caiman_asr_train.rnnt.loss import TransducerLossFn
from caiman_asr_train.rnnt.model import RNNT


class Encoder(torch.nn.Module):
    """
    View exposing only the encoder's parameters (disjoint from joint/pred).
    """

    def __init__(self, model: RNNT):
        # Register only the parameters encode() uses, see joint comment
        super().__init__()
        self.pre_rnn = model.encoder["pre_rnn"]
        self.post_rnn = model.encoder["post_rnn"]
        if "prefix_embd" in model.encoder:
            self.prefix_embd = model.encoder["prefix_embd"]
        self.joint_enc = model.joint_enc
        self._encode = model.encode

    def forward(self, *, state, prefix, feats):
        return self._encode(state=state, prefix=prefix, feats=feats)


class Prediction(torch.nn.Module):
    """
    View exposing only the prediction network's parameters (disjoint from enc/joint).
    """

    def __init__(self, model: RNNT):
        # Register only the parameters predict() uses, see joint comment
        super().__init__()
        self.embed = model.prediction["embed"]
        self.dec_rnn = model.prediction["dec_rnn"]
        self.joint_pred = model.joint_pred
        self._predict = model.predict

    def forward(self, *, state, prefix, tokens):
        return self._predict(state=state, prefix=prefix, tokens=tokens)


@beartype
@dataclass
class RNNTSubModels:
    """
    Wrapper class to access sub-models (encoder, pred, joint+loss) of an RNNT.

    Under the hood there is a single RNNT model object that is used in the various
    sub-models. The Encoder/Prediction/JointLoss views each register ONLY the parameters
    their own forward uses (a disjoint partition of the RNNT's parameters); they hold the
    RNNT via a bound method rather than as a registered submodule, so the full parameter
    set is not re-registered. This means each parameter is owned by exactly one view, so
    wrapping the views in DistributedDataParallel gives one Reducer per parameter with no
    "unused" parameters -- no ``static_graph`` is required and there is no shared-module
    gradient-routing ambiguity.

    This class copies some of the the torch.nn.Module methods for ease of use
    (e.g. train/eval) but does not support the full set of methods. Feel free to add
    missing ones as is useful. This class does not inherit from torch.nn.Module as
    then the model's parameters would be registered to multiple modules which could
    cause confusion and bugs. state_dict / load_state_dict / parameters delegate to the
    underlying RNNT, so the saved checkpoint is unaffected by the views.
    """

    encoder: Union[DDP, Encoder]
    prediction: Union[DDP, Prediction]
    joint_loss: Union[DDP, JointLoss]
    rnnt: RNNT

    def __post_init__(self):
        expect = Counter(map(id, self.rnnt.parameters()))
        triple = Counter(
            id(p)
            for view in (self.encoder, self.prediction, self.joint_loss)
            for p in view.parameters()
        )
        assert triple == expect, (
            "RNNTSubModels views must 1-to-1 partition the RNNT parameters; "
            f"{sum((expect - triple).values())} unowned (never all-reduced), "
            f"{sum((triple - expect).values())} over-owned (foreign or all-reduced twice)."
        )

    @classmethod
    def from_RNNT(
        cls,
        model: RNNT,
        ddp: bool = False,
        *,
        loss_fn: TransducerLossFn,
        dropout_prob: float = 0.0,
    ):
        encoder, pred = Encoder(model), Prediction(model)
        joint_loss = build_joint_loss(model, loss_fn, dropout_prob)

        if ddp:
            current_device = torch.cuda.current_device()
            joint_loss = DDP(joint_loss, device_ids=[current_device])
            torch.distributed.barrier()
            encoder = DDP(encoder, device_ids=[current_device])
            torch.distributed.barrier()
            pred = DDP(pred, device_ids=[current_device])

        return cls(encoder, pred, joint_loss, model)

    @property
    def config(self):
        return self.rnnt.config

    def train(self, mode=True):
        self.rnnt.train(mode)
        # The Fused JointLoss kept a separate TransducerJoint (which does dropout)
        # hence need an extra mode switch.
        self.joint_loss.train(mode)

    def eval(self):
        self.train(False)

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        return self.rnnt.load_state_dict(state_dict, strict)

    def state_dict(self, *args, **kwargs):
        return self.rnnt.state_dict(*args, **kwargs)

    def named_parameters(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[Tuple[str, Parameter]]:
        return self.rnnt.named_parameters(prefix, recurse)

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        return self.rnnt.parameters(recurse)
