import glob
import hashlib
import os
from functools import cache

from beartype import beartype

from caiman_asr_train.tilelstm.tree_model import TRAIN_DATA_GLOB, read_data_dir

# Cache-root-relative path to the trained bundle; cache_read/cache_write add the root.
BUNDLE_REL = os.path.join("caiman_lstm_tree", "model_bundle.joblib")


# Source files (tilelstm-relative) whose contents determine the trained bundle: the
# featurization + fitted models, plus the autotune modules featurize imports (outside
# tree_model/, so a package walk would miss them). The fingerprint hashes these + the
# training data; extend it when featurization or training grows a new source dependency.
_FEATURE_SOURCE_FILES = (
    "tree_model/__init__.py",  # BACKENDS: which per-backend models the bundle holds
    "tree_model/core/features.py",  # the feature vectors themselves
    "tree_model/core/specs.py",  # recorded GpuSpec scalars featurize evaluates on
    "tree_model/core/train.py",  # training + bundle assembly
    "autotune/configs_smem.py",  # per-config SMEM-byte feature
    "autotune/kernel.py",  # Kernel.is_* predicates featurize / BACKENDS rely on
)


@cache
@beartype
def source_data_fingerprint() -> str:
    """
    SHA-256 over the source files + training data that determine the trained bundle, so a
    stale cached bundle is rebuilt automatically when featurization, training, or the data
    changes. Hashes each input's label + bytes in sorted-label order; source files are
    labelled by their ``tilelstm``-relative path, the sweeps by basename -- so the same
    data hashes identically whether read in-tree or from the datasets drive (whose
    absolute location varies with the checkout/mount). ``@cache``d since the inputs don't
    change mid-process.
    """
    # __file__ is tree_model/core/fingerprint.py; three levels up is the tilelstm root the
    # source paths are relative to.
    tilelstm = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    inputs = [(rel, os.path.join(tilelstm, rel)) for rel in _FEATURE_SOURCE_FILES]
    # The training sweeps, from the same resolved dir train reads (out-of-tree datasets
    # location when populated, else the in-tree data/) -- so the bundle is gated on the
    # data that actually trained it.
    inputs += [
        (os.path.basename(path), path)
        for path in glob.glob(os.path.join(read_data_dir(), TRAIN_DATA_GLOB))
    ]

    h = hashlib.sha256()
    for label, path in sorted(inputs):
        h.update(label.encode())
        h.update(b"\0")
        with open(path, "rb") as fh:
            h.update(fh.read())
        h.update(b"\0")
    return h.hexdigest()
