#!/usr/bin/env python3

from argparse import Namespace

import torch
from beartype import beartype
from beartype.typing import Tuple

from rnnt_train.setup.base import CPU, PipelineType
from rnnt_train.setup.val import BaseValSetup


@beartype
class ValCPUSetup(BaseValSetup):
    def modify_dataloader_kw(
        self,
        args: Namespace,
        dataset_kw: dict,
        features_kw: dict,
        pipeline_type: PipelineType,
    ):
        if args.dump_nth != None:
            features_kw["dither"] = 0.0

    def preferred_device(self) -> torch.device:
        return CPU

    def start_ddp(self, args) -> Tuple[bool, int]:
        return False, 1

    def use_torch_dist(self, args: Namespace) -> bool:
        return False
