import os

import numpy as np
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline

# Copyright (c) 2022 Myrtle.ai
# written by rob@myrtle, Jun 2022


def extract(sample_rate, n_fft, mel_dim):
    """
    This function extracts the implicit fft-to-mel transform matrix used by DALI by feeding
    the DALI algorithm a series of contrived inputs designed to extract the matrix columns.
    """
    # half plus one
    hpo = n_fft // 2 + 1

    # mkdir /tmp/exmats
    if not os.path.exists("/tmp/exmats"):
        os.mkdir("/tmp/exmats")

    # each vector here pulls the ith col from the 'real' melmat inside DALI
    with open("/tmp/exmat_file_list.txt", "w") as f:
        for i in range(hpo):
            extractcol = np.zeros((hpo, 1), dtype=np.float32)
            extractcol[i] = 1.0
            fn = f"/tmp/exmats/excol{i}.npy"
            np.save(fn, extractcol)
            _ = f.write(fn)
            _ = f.write("\n")

    root_dir = ""
    batch_size = hpo

    class SimplePipeline(Pipeline):
        def __init__(self, batch_size, num_threads, device_id):
            super(SimplePipeline, self).__init__(
                batch_size, num_threads, device_id, seed=12
            )
            self.input = ops.NumpyReader(
                file_root=root_dir, file_list="/tmp/exmat_file_list.txt"
            )
            self.mel = ops.MelFilterBank(
                sample_rate=sample_rate, nfilter=mel_dim, normalize=True
            )
            #

        def define_graph(self):
            data = self.input()
            mels = self.mel(data)
            return (data, mels)

    pipe = SimplePipeline(batch_size, 1, 0)
    pipe.build()

    pipe_out = pipe.run()

    contrived, extracts = pipe_out

    contrived_tensor = contrived.as_tensor()
    extracts_tensor = extracts.as_tensor()

    daliextracts = np.array(extracts_tensor)
    extractmat = daliextracts[:, :, 0].T

    return extractmat
