"""Tests for register_dataclass_as_pytree (core/pytree.py).

These guard that DDP / torch pytree can find the tensors inside the batch/state
dataclasses: tensor fields must flatten to leaves, scalar fields must be carried as
static metadata (not leaves), nested registered dataclasses must be recursed into, and a
nested *unregistered* dataclass must raise rather than silently hide its tensors.

CPU-only -- runs in the normal test suite (no GPU / torchrun needed).
"""

from dataclasses import dataclass

import pytest
import torch
from torch.utils._pytree import tree_flatten, tree_unflatten

from caiman_asr_train.core.pytree import register_dataclass_as_pytree


@register_dataclass_as_pytree
@dataclass
class _PureTensor:
    a: torch.Tensor
    b: torch.Tensor


@register_dataclass_as_pytree
@dataclass
class _Mixed:
    data: torch.Tensor
    other: torch.Tensor
    factor: int
    name: str
    maybe: object


@register_dataclass_as_pytree
@dataclass
class _Outer:
    inner: _PureTensor
    feat: torch.Tensor
    tag: str


@dataclass
class _Unregistered:
    hidd: torch.Tensor


@register_dataclass_as_pytree
@dataclass
class _BadNesting:
    inner: object


def test_pure_tensor_dataclass_roundtrips():
    a, b = torch.randn(2, 3), torch.randn(4)
    obj = _PureTensor(a=a, b=b)
    leaves, spec = tree_flatten(obj)
    assert len(leaves) == 2 and leaves[0] is a and leaves[1] is b
    rebuilt = tree_unflatten(leaves, spec)
    assert rebuilt.a is a and rebuilt.b is b


def test_mixed_dataclass_flattens_only_tensors():
    """Non-tensor fields are static metadata (treespec context), not pytree leaves."""
    a, b = torch.randn(2, 3, requires_grad=True), torch.randn(4)
    obj = _Mixed(data=a, other=b, factor=7, name="lang", maybe=None)

    leaves, spec = tree_flatten(obj)

    # Only the two tensors are leaves; factor / name / maybe are carried in the context.
    assert len(leaves) == 2
    assert leaves[0] is a and leaves[1] is b
    assert all(torch.is_tensor(leaf) for leaf in leaves)

    # Roundtrip rebuilds the tensors and the static metadata.
    rebuilt = tree_unflatten(leaves, spec)
    assert rebuilt.data is a and rebuilt.other is b
    assert rebuilt.factor == 7
    assert rebuilt.name == "lang"
    assert rebuilt.maybe is None


def test_mixed_metadata_is_part_of_treespec():
    """Differing scalar metadata yields differing treespecs (metadata is structural)."""
    a, b = torch.randn(2), torch.randn(2)
    spec1 = tree_flatten(_Mixed(a, b, 1, "en", None))[1]
    spec2 = tree_flatten(_Mixed(a, b, 2, "en", None))[1]
    spec_same = tree_flatten(_Mixed(a, b, 1, "en", None))[1]
    assert spec1 != spec2
    assert spec1 == spec_same


def test_nested_registered_dataclass_is_recursed():
    inner = _PureTensor(a=torch.randn(2, 2, requires_grad=True), b=torch.randn(2, 2))
    feat = torch.randn(3, requires_grad=True)
    obj = _Outer(inner=inner, feat=feat, tag="x")

    leaves, spec = tree_flatten(obj)

    # The nested dataclass is recursed into: its two tensors plus feat are all leaves.
    assert len(leaves) == 3
    assert all(torch.is_tensor(leaf) for leaf in leaves)
    assert {id(leaf) for leaf in leaves} == {id(inner.a), id(inner.b), id(feat)}

    rebuilt = tree_unflatten(leaves, spec)
    assert isinstance(rebuilt.inner, _PureTensor)
    assert rebuilt.inner.a is inner.a and rebuilt.inner.b is inner.b
    assert rebuilt.feat is feat
    assert rebuilt.tag == "x"


def test_unregistered_nested_dataclass_raises():
    bad = _BadNesting(inner=_Unregistered(hidd=torch.randn(2)))
    with pytest.raises(TypeError, match="not registered as a pytree node"):
        tree_flatten(bad)
