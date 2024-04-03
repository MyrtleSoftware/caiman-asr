from dataclasses import dataclass

import torch
from beartype import beartype
from beartype.typing import Any, Iterator, Mapping, Tuple, Union
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parameter import Parameter

from caiman_asr_train.rnnt.model import RNNT


class Joint(torch.nn.Module):
    def __init__(self, model: RNNT):
        super().__init__()
        self.model = model

    def forward(self, f, g, f_len, g_len, batch_offset):
        return self.model.joint(f, g, f_len, g_len, batch_offset)


class Encoder(torch.nn.Module):
    def __init__(self, model: RNNT):
        super().__init__()
        self.model = model

    def forward(self, x, x_lens, enc_state):
        return self.model.encode(x, x_lens, enc_state)


class Prediction(torch.nn.Module):
    def __init__(self, model: RNNT):
        super().__init__()
        self.model = model

    def forward(self, y, pred_state, add_sos, special_sos):
        return self.model.predict(y, pred_state, add_sos, special_sos)


@beartype
@dataclass
class RNNTSubModels:
    """
    Wrapper class to access sub-models (encoder, pred, joint) of an RNNT.

    Under the hood there is a single RNNT model object that is used in the various
    sub-models. This class assumes that:

    1) There are no parameters/buffers shared between the sub-models.
    2) The initial RNNT class passed in the from_RNNT method is the single source of
    truth for maintaining these buffers/parameters. The wrapper classes
    Encoder/Joint/Prediction are just views on the RNNT model's parameters.

    This class copies some of the the torch.nn.Module methods for ease of use
    (e.g. train/eval) but does not support the full set of methods. Feel free to add
    missing ones as is useful. This class does not inherit from torch.nn.Module as
    then the model's parameters would be registered to multiple modules which could
    cause confusion and bugs.
    """

    encoder: Union[DDP, Encoder]
    joint: Union[DDP, Joint]
    prediction: Union[DDP, Prediction]
    rnnt: RNNT

    @classmethod
    def from_RNNT(cls, model: RNNT, ddp: bool = False):
        encoder, joint, pred = Encoder(model), Joint(model), Prediction(model)
        if not ddp:
            return cls(encoder, joint, pred, model)

        current_device = torch.cuda.current_device()
        joint = DDP(joint, device_ids=[current_device], static_graph=True)
        torch.distributed.barrier()
        encoder = DDP(encoder, device_ids=[current_device], static_graph=True)
        torch.distributed.barrier()
        pred = DDP(pred, device_ids=[current_device], static_graph=True)

        return cls(encoder, joint, pred, model)

    def train(self, mode=True):
        self.rnnt.train(mode)

    def eval(self):
        self.rnnt.eval()

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
