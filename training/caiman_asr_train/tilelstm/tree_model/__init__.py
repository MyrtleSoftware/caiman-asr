"""
Learned config-ranking heuristic for the TileLang LSTM autotuner.
"""

import glob
import os

from beartype import beartype

from caiman_asr_train.tilelstm.autotune.kernel import Kernel

# In-tree per-GPU training sweeps: gen_data's default WRITE target, and the development
# READ fallback (see read_data_dir).
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Canonical out-of-tree home of the finished sweeps inside the container (the mounted
# datasets drive). read_data_dir() prefers it whenever it holds training sweeps.
EXTERNAL_DATA_DIR = "/datasets/tile_tree_data"

# Sweeps train fits on and fingerprint hashes; excludes the plot-only custom_lstm_*
# baselines so they never force a retrain.
TRAIN_DATA_GLOB = "full_ranked_*.jsonl.gz"


@beartype
def read_data_dir() -> str:
    """
    The directory sweep data is READ from (training, the bundle fingerprint, and the
    plots' default): ``EXTERNAL_DATA_DIR`` when it contains at least one training sweep
    (the production location), else the in-tree ``DATA_DIR`` (the development fallback,
    which is also ``gen_data``'s default write target -- see ``data/README.md``).
    """
    if glob.glob(os.path.join(EXTERNAL_DATA_DIR, TRAIN_DATA_GLOB)):
        return EXTERNAL_DATA_DIR
    return DATA_DIR


# Backend order (cooperative, persistent, split-K last).
BACKENDS = (
    Kernel.fwd_c,
    Kernel.fwd_ct,
    Kernel.bwd_c,
    Kernel.fwd_p,
    Kernel.fwd_pt,
    Kernel.bwd_p,
    Kernel.bwd_s,
)
