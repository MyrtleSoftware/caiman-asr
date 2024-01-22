# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch


def lr_policy(
    optimizer,
    initial_lr,
    min_lr,
    step,
    warmup_steps,
    hold_steps,
    half_life_steps,
    dist_lamb=False,
):
    """
    learning rate policy
    Args:
        optimizer: optimizer
        initial_lr: base learning rate
        min_lr: minimum learning rate
        step: current iteration (update) number
        warmup_steps: number of steps over which learning rate ramps up
        hold_steps: number of steps over which learning rate is constant
        half_life_steps: learning rate half-life in steps during decay
        dist_lamb: If True, we are using the DistributedFusedLAMB optimizer.
    """

    if step < warmup_steps:
        a = (step + 1) / (warmup_steps + 1)
    elif step < warmup_steps + hold_steps:
        a = 1.0
    else:
        a = 0.5 ** ((step - warmup_steps - hold_steps) / half_life_steps)

    if type(initial_lr) is float:
        initial_lr = [initial_lr]
    assert len(initial_lr) == len(optimizer.param_groups)

    if not dist_lamb:
        for lr, param_group in zip(initial_lr, optimizer.param_groups):
            param_group["lr"] = max(a * lr, min_lr)
    else:
        for lr, sub_optim in zip(initial_lr, optimizer.child_optimizers):
            sub_optim._lr = max(a * lr, min_lr * torch.ones_like(lr))
