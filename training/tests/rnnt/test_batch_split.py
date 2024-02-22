import copy
from argparse import Namespace

import pytest
import torch
from beartype.typing import Union
from torch.nn.parallel import DistributedDataParallel as DDP

from rnnt_train.common.batch_splitting import train_step_batch_split
from rnnt_train.common.helpers import unwrap_ddp
from rnnt_train.common.seed import set_seed
from rnnt_train.common.train_aux import train_step
from rnnt_train.rnnt.loss import ApexTransducerLoss
from rnnt_train.rnnt.model import RNNT
from rnnt_train.rnnt.sub_models import RNNTSubModels


@pytest.fixture()
def loss_fn(n_classes_fixture):
    return ApexTransducerLoss(
        blank_idx=n_classes_fixture - 1,
        packed_input=False,
        validate_first_n_remaining=0,
    )


@pytest.fixture()
def args():
    return Namespace(grad_accumulation_batches=1, batch_split_factor=8, num_gpus=1)


@pytest.fixture()
def device():
    return "cuda"


@torch.no_grad()
@pytest.fixture()
def input_data_factory(input_dim_fixture, n_classes_fixture, device):
    def _gen_data(seed=0, local_rank=0):
        set_seed(seed=seed, local_rank=local_rank)

        audio_seq_len = 20
        txt_seq_len = 10
        batch_size = 8

        feats = torch.randn(audio_seq_len, batch_size, input_dim_fixture, device=device)
        # No padding:
        feat_lens = torch.tensor([audio_seq_len] * batch_size, device=device)
        txt = torch.randint(
            0,
            n_classes_fixture - 1,
            (batch_size, txt_seq_len),
            device=device,
        )
        # No padding:
        txt_lens = torch.tensor(
            [txt_seq_len] * batch_size, device=device, dtype=torch.int
        )
        return feats, feat_lens, txt, txt_lens

    return _gen_data


def build_objects(
    model_factory,
    input_data_factory,
    args,
    device,
    no_amp,
    local_rank=0,
):
    """
    Create identical pair of (models, scaler, data) objects for batch splitting on & off.

    Note that the models are not identical as the batch splitting code requires the
    RNNTSubModels class to be used but the weights are the same and the forward
    (and backwards) passes should be equivalent.
    """
    ddp = torch.distributed.is_initialized()

    model = model_factory(local_rank=local_rank).to(device)
    model_copy = copy.deepcopy(model)

    sub_model = RNNTSubModels.from_RNNT(model_copy, ddp=ddp)
    if ddp:
        model = DDP(model, device_ids=[torch.cuda.current_device()])
        args.num_gpus = torch.distributed.get_world_size()

    args.no_amp = no_amp
    scaler1 = torch.cuda.amp.GradScaler() if not no_amp else None
    # when running distributed the data on each rank should be different
    data1 = input_data_factory(local_rank=local_rank)
    scaler2, data2 = (copy.deepcopy(scaler1), copy.deepcopy(data1))

    return (sub_model, scaler2, data2), (model, scaler1, data1)


@pytest.mark.parametrize(
    "no_amp",
    [
        True,
        pytest.param(
            False,
            marks=pytest.mark.xfail(
                reason="With amp on, the test fails due to numerical differences"
            ),
        ),
    ],
)
def test_batch_split_train_step_equiv(
    model_factory, loss_fn, input_data_factory, args, device, no_amp
):
    """
    Test forward passes for train_step and train_step_batch_split are equivalent.
    """
    (model1, scaler1, data1), (model2, scaler2, data2) = build_objects(
        model_factory, input_data_factory, args, device, no_amp
    )
    results1 = train_step_batch_split(
        model1,
        loss_fn,
        args,
        *data1,
        scaler=scaler1,
        rnnt_state=None,
    )

    results2 = train_step(
        model2,
        loss_fn,
        args,
        *data2,
        scaler=scaler2,
        rnnt_state=None,
    )

    compare_step_results(results1, results2, model1, model2, scaler1, scaler2)


ALLOW_FAIL_200 = pytest.param(
    200,
    marks=pytest.mark.xfail(
        reason="NaNs in the loss and the weights occurs when n_steps > 150"
    ),
)


@pytest.mark.parametrize("train_step_fn", [train_step, train_step_batch_split])
@pytest.mark.parametrize(
    "n_steps",
    [1, 10, ALLOW_FAIL_200],
)
def test_dist_equiv_optim_steps(
    model_factory,
    optimizer_factory,
    loss_fn,
    input_data_factory,
    args,
    device,
    train_step_fn,
    n_steps,
):
    if not torch.distributed.is_initialized():
        pytest.skip("Skipping distributed test")
    local_rank = torch.distributed.get_rank()
    args.no_amp = True
    model = model_factory(local_rank=local_rank).to(device)
    optimizer = optimizer_factory(model)
    if train_step_fn == train_step:
        model = DDP(model, device_ids=[torch.cuda.current_device()])
    else:
        model = RNNTSubModels.from_RNNT(model, ddp=True)

    args.num_gpus = torch.distributed.get_world_size()

    # after using DDP the model weights should be the same across ranks
    assert_model_equal_on_all_ranks(model, local_rank, check_grads=False)

    for i in range(n_steps):
        # The data should be different for each rank
        input_data = input_data_factory(seed=i * 10, local_rank=local_rank)
        optimizer.zero_grad()
        _, loss_nan, _ = train_step_fn(
            model,
            loss_fn,
            args,
            *input_data,
            scaler=None,
            rnnt_state=None,
        )
        if loss_nan:
            continue
        assert_model_equal_on_all_ranks(model, local_rank, check_grads=True)
        optimizer.step()

    assert_model_equal_on_all_ranks(model, local_rank, check_grads=False)


def assert_model_equal_on_all_ranks(
    model: Union[DDP, RNNTSubModels], local_rank, check_grads
):
    """
    Assert the model weights or grads are the same across all ranks.

    All ranks >0 send their model tensors to rank 0. It would be possible to use a ring
    topology to reduce the number of recvs on rank 0 but it's easier to debug this way.
    """
    if local_rank == 0:
        for rank in range(1, torch.distributed.get_world_size()):
            receive_and_compare_model(model, rank, check_grads)
    else:
        send_model(model, local_rank, check_grads)
    torch.distributed.barrier()


def send_model(model, local_rank, check_grads):
    for name, param in model.named_parameters():
        send = param if not check_grads else param.grad
        torch.distributed.send(send, dst=0, tag=local_rank)


def receive_and_compare_model(model, rank, check_grads):
    params_wrong = []
    for name, param in model.named_parameters():
        local = param if not check_grads else param.grad
        received_param = torch.empty_like(local)
        torch.distributed.recv(received_param, src=rank, tag=rank)
        try:
            assert torch.allclose(local, received_param, atol=1e-4, rtol=1e-3)
        except AssertionError:
            print(f"Rank {rank} model received and compared")
            print(f"name: {name}")
            params_wrong.append(name)
    assert not params_wrong, (
        f"Rank {rank} {check_grads=} param check failed for {len(params_wrong)} "
        f"params: {params_wrong}"
    )
    print(f"Rank {rank} model received and compared")


def compare_step_results(
    results1,
    results2,
    model1: RNNTSubModels,
    model2: Union[RNNT, DDP],
    scaler1,
    scaler2,
):
    """
    Assert that results1 and results2 are approximately equal.
    """
    loss_item, loss_nan, rnnt_state = results1
    loss_item2, loss_nan2, rnnt_state2 = results2

    assert pytest.approx(loss_item) == loss_item2
    assert loss_nan == loss_nan2

    model2 = unwrap_ddp(model2)
    params2 = {k: v for k, v in model2.named_parameters()}
    grad_failed = []
    for k, v in model1.named_parameters():
        assert torch.allclose(v, params2[k])
        if scaler1 is not None:
            assert scaler1.get_scale() == scaler2.get_scale()
        try:
            # fill inf grads with 0
            mask_inf = v.grad.isinf() | params2[k].grad.isinf()
            v.grad.masked_fill_(mask_inf, 0)
            params2[k].grad.masked_fill_(mask_inf, 0)

            assert torch.allclose(
                v.grad, params2[k].grad, rtol=1e-4, atol=1e-4
            ), f"{k} grad mismatch"
        except AssertionError:
            grad_failed.append(k)
    assert (
        not grad_failed
    ), f"Grad check failed for {len(grad_failed)} params: {grad_failed}"


if __name__ == "__main__":
    import os
    from argparse import ArgumentParser

    import torch.distributed.run as distrib_run

    parser = ArgumentParser()
    parser.add_argument("--called_by_torchrun", action="store_true")
    parser.add_argument("--num_gpus", type=int, default=2)
    parser.add_argument(
        "--local_rank",
        default=os.getenv("LOCAL_RANK", 0),
        type=int,
        help="GPU id used for distributed training",
    )

    parsed_args = parser.parse_args()

    if not parsed_args.called_by_torchrun:
        torchrun_parser = distrib_run.get_args_parser()
        torchrun_args = torchrun_parser.parse_args(
            [
                "--standalone",
                "--nnodes",
                "1",
                "--nproc_per_node",
                str(parsed_args.num_gpus),
                "tests/rnnt/test_batch_split.py",
                "--called_by_torchrun",
            ]
        )
        distrib_run.run(torchrun_args)
        exit()

    # We are running with torchrun, so we should run the tests
    # and exit
    local_rank = os.getenv("LOCAL_RANK", 0)
    torch.cuda.set_device(parsed_args.local_rank)

    torch.distributed.init_process_group(
        backend="nccl",
        init_method="env://",
    )
    pytest.main([__file__])
    exit()
