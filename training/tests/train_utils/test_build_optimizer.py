from argparse import Namespace

from apex.optimizers import FusedLAMB

from caiman_asr_train.train_utils.build_optimizer import build_optimizer


def test_build_optimizer(mini_model_factory):
    model, _ = mini_model_factory()
    args = Namespace(lr=0.001, weight_decay=0.01, beta1=0.9, beta2=0.999, clip_norm=1.0)
    optimizer = build_optimizer(args, model)
    assert isinstance(optimizer, FusedLAMB)
