#!/usr/bin/env python3
import sys

import torch.distributed.run as distrib_run
from beartype import beartype


@beartype
def maybe_restart_with_torchrun(
    num_gpus: int, already_called_by_torchrun: bool, script: str
) -> None:
    if num_gpus > 1 and not already_called_by_torchrun:
        # Call this script again, this time with torchrun
        # to use multiple GPUs
        torchrun_parser = distrib_run.get_args_parser()
        torchrun_args = torchrun_parser.parse_args(
            [
                "--standalone",
                "--nnodes",
                "1",
                "--nproc_per_node",
                str(num_gpus),
                script,
                "--called_by_torchrun",
            ]
            + sys.argv[1:]
        )
        distrib_run.run(torchrun_args)
        # At this point the script has run successfully on multiple GPUs,
        # so shouldn't continue this thread
        sys.exit()
