"""
Register the core batch/state dataclasses as PyTorch pytree nodes.

PyTorch locates the tensors inside a module's forward output by flattening it with
``torch.utils._pytree.tree_flatten`` -- this is how ``DistributedDataParallel`` finds the
tensors whose gradients it must all-reduce (and how ``torch.compile`` traces I/O).
``tree_flatten`` only decomposes *registered* container types; an unregistered
``@dataclass`` is treated as a single opaque leaf, so any tensor nested inside it is
invisible to the traversal.

Registering the dataclasses as pytree nodes makes the inner tensors visible again. See
pytorch/pytorch#106690 and huggingface/transformers#25357 for the identical failure mode
with ``OrderedDict`` / ``ModelOutput`` subclasses.
"""

import dataclasses

import torch
from torch.utils._pytree import register_pytree_node

# Classes registered via register_dataclass_as_pytree. Lets us distinguish a *registered*
# nested dataclass (recurse into it) from an unregistered one (whose tensors would be
# silently hidden -- we raise instead of mis-handling it).
_REGISTERED: set = set()


def _is_static_metadata(value) -> bool:
    """
    Whether a dataclass field value is static treespec metadata (not a pytree child).

    Dynamic children (returned False), recursed into so their tensors are visible to DDP /
    ``tree_map``: tensors, standard containers (list/tuple/dict) and *registered* nested
    dataclasses. Static metadata (returned True), carried in the treespec context: plain
    scalars such as int, float, bool, str, None, enums. A nested dataclass that is *not*
    registered would hide its tensors, so we raise rather than mis-handle it.
    """
    if torch.is_tensor(value) or isinstance(value, (list, tuple, dict)):
        return False

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        if type(value) in _REGISTERED:
            return False

        raise TypeError(
            f"Cannot flatten field of type {type(value).__name__!r}: it is a "
            f"dataclass that is not registered as a pytree node, so its tensors "
            f"would be invisible to DDP. Decorate it with @register_dataclass_as_pytree."
        )

    return True


def register_dataclass_as_pytree(cls):
    """
    Class decorator that registers a ``@dataclass`` as a pytree node.

    Tensor / container / registered-dataclass fields become pytree *children* (recursed
    into, mapped over, found by DDP); scalar "metadata" fields are carried as static
    treespec context (so they are not mistaken for leaves). Handles nested registered
    dataclasses and mixed tensor/non-tensor dataclasses, and raises on a nested dataclass
    that is not itself registered. Idempotent: re-registering a class is a no-op.
    """
    assert dataclasses.is_dataclass(
        cls
    ), f"Can only decorate a @dataclass, not {cls.__name__!r}"

    field_names = [f.name for f in dataclasses.fields(cls)]

    _REGISTERED.add(cls)

    def flatten(obj):
        children, child_names, static = [], [], []
        for name in field_names:
            value = getattr(obj, name)
            if _is_static_metadata(value):
                static.append((name, value))
            else:
                children.append(value)
                child_names.append(name)

        return children, (tuple(child_names), tuple(static))

    def unflatten(children, context):
        child_names, static = context
        kwargs = dict(zip(child_names, children))
        kwargs.update(static)
        return cls(**kwargs)

    try:
        register_pytree_node(cls, flatten, unflatten)
    except ValueError:
        # Already registered -- registration is process-global and idempotent.
        pass

    return cls
