"""
The `Kernel` enum: the single identity for each of the seven TileLang LSTM scan kernels.

`Kernel` is a `str`-subclass enum, so a member IS its string value: it compares equal to
and hashes like its value, and json-serialises to it. That keeps the serialization /
dict-key boundaries transparent -- the on-disk tune `cache_key`, the tree-model JSONL
`backend` field, and the model-bundle keys all round-trip with zero special-casing --
while code uses the typed member (`Kernel.fwd_c`) and its behaviour methods.

Naming: ``<family-prefix>_<variant>`` -- fwd_* are forward (inference), the _t* train
variants also save gates, bwd_* are the recurrent backward. c = cooperative grid,
p = persistent (fixed bps*sms launch), s = split-K. The values equal the names.
"""

from enum import Enum


class Family(Enum):
    """
    The call contract a kernel obeys -- which buffers, reference, and per-config
    validation the tuner drives it with. ``fwd`` and ``fwd_train`` share the forward-scan
    buffers (train additionally saves the gates ``As``); ``bwd`` is the recurrent backward.
    Plain ``Enum`` (not serialized -- it never reaches the cache_key / JSONL boundaries that
    make ``Kernel`` a str-subclass), used only for typed dispatch in the tuner.
    """

    fwd = "fwd"
    fwd_train = "fwd_train"
    bwd = "bwd"


class Kernel(str, Enum):
    fwd_c = "fwd_c"  # cooperative forward (inference)
    fwd_p = "fwd_p"  # persistent forward (inference)
    fwd_ct = "fwd_ct"  # cooperative forward, gate-saving
    fwd_pt = "fwd_pt"  # persistent forward, gate-saving
    bwd_c = "bwd_c"  # cooperative backward
    bwd_s = "bwd_s"  # split-K backward (small batch)
    bwd_p = "bwd_p"  # persistent backward

    def __str__(self) -> str:
        # Clean logs/messages -- "fwd_c", not "Kernel.fwd_c".
        return self.value

    # -- variant predicates --------------------------------------------------------------
    def is_coop(self) -> bool:
        """Cooperative-grid kernel (fwd_c / fwd_ct / bwd_c)."""
        return self in _COOP

    def is_persistent(self) -> bool:
        """Persistent (fixed bps*sms launch) kernel (fwd_p / fwd_pt / bwd_p)."""
        return self in _PERSISTENT

    def is_splitk(self) -> bool:
        """The split-K backward kernel (bwd_s)."""
        return self is Kernel.bwd_s

    # -- direction / mode ----------------------------------------------------------------
    def is_forward(self) -> bool:
        """A forward-scan kernel (any fwd_*, incl. the gate-saving train variants)."""
        return self in _FORWARD

    def is_backward(self) -> bool:
        """A recurrent-backward kernel (any bwd_*)."""
        return self in _BACKWARD

    def is_train(self) -> bool:
        """A gate-saving (training) forward (fwd_ct / fwd_pt)."""
        return self in _TRAIN

    @property
    def family(self) -> Family:
        """The call-contract ``Family`` driving buffers + reference + validation
        (``Family.fwd`` / ``Family.fwd_train`` / ``Family.bwd``)."""
        if self.is_backward():
            return Family.bwd
        return Family.fwd_train if self.is_train() else Family.fwd


_COOP = frozenset({Kernel.fwd_c, Kernel.fwd_ct, Kernel.bwd_c})
_PERSISTENT = frozenset({Kernel.fwd_p, Kernel.fwd_pt, Kernel.bwd_p})
_FORWARD = frozenset({Kernel.fwd_c, Kernel.fwd_p, Kernel.fwd_ct, Kernel.fwd_pt})
_BACKWARD = frozenset({Kernel.bwd_c, Kernel.bwd_s, Kernel.bwd_p})
_TRAIN = frozenset({Kernel.fwd_ct, Kernel.fwd_pt})
