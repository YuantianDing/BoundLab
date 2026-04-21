"""Tests for ``boundlab.utils.multiple_diagnonal`` /
``multiple_diag_embed`` and the ``EinsumOp.add_conditions`` /
``remove_conditions`` helpers that build on them.
"""

import itertools

import pytest
import torch

from boundlab.linearop import EinsumOp, SumOp
from boundlab.utils import EQCondition, multiple_diag_embed, multiple_diagnonal


def _row_norms_from_jacobian(jac, p, out_shape):
    """Ground-truth per-output-element p-norm of a Jacobian row."""
    out_n = out_shape.numel()
    return jac.reshape(out_n, -1).norm(p, dim=1).reshape(out_shape)


# ---------------------------------------------------------------------------
# multiple_diagnonal
# ---------------------------------------------------------------------------


def test_multiple_diagnonal_2d_pair():
    t = torch.arange(9.).reshape(3, 3)
    out, dim_map = multiple_diagnonal(t, [(0, 1)])
    assert out.tolist() == [0., 4., 8.]
    # Both original dims 0 and 1 collapsed into the (new) trailing dim 0.
    assert dim_map == [0, 0]


def test_multiple_diagnonal_3d_pair_keeps_spectator():
    t = torch.arange(2 * 3 * 3.).reshape(2, 3, 3)
    out, dim_map = multiple_diagnonal(t, [(1, 2)])
    # diagonal(dim1=1, dim2=2) drops dims (1, 2) and appends a size-3 trailing dim.
    assert out.shape == (2, 3)
    # Spectator dim 0 stays at position 0; dims 1&2 both point to trailing dim 1.
    assert dim_map == [0, 1, 1]
    expected = torch.stack([t[0].diagonal(), t[1].diagonal()])
    assert torch.equal(out, expected)


def test_multiple_diagnonal_two_pairs():
    t = torch.arange(3 * 3 * 4 * 4.).reshape(3, 3, 4, 4)
    out, dim_map = multiple_diagnonal(t, [(0, 1), (2, 3)])
    # Both pairs are collapsed; both original-pairs land on trailing dims.
    assert out.shape == (3, 4)  # first diag → [4,4,3] then diag(0,1) → [3,4]
    # Values: out[i, j] == t[i, i, j, j]
    for i, j in itertools.product(range(3), range(4)):
        assert out[i, j].item() == t[i, i, j, j].item()
    # The two pairs should map to distinct positions.
    assert dim_map[0] == dim_map[1]
    assert dim_map[2] == dim_map[3]
    assert dim_map[0] != dim_map[2]


def test_multiple_diagnonal_non_adjacent_pair():
    t = torch.arange(3 * 5 * 3.).reshape(3, 5, 3)
    out, dim_map = multiple_diagnonal(t, [(0, 2)])
    # diagonal(0, 2) removes dims 0 and 2, appends new trailing dim of size 3.
    # Surviving original dim 1 moves to position 0; dims 0/2 map to 1.
    assert out.shape == (5, 3)
    assert dim_map == [1, 0, 1]
    for i, j in itertools.product(range(3), range(5)):
        assert out[j, i].item() == t[i, j, i].item()


# ---------------------------------------------------------------------------
# multiple_diag_embed
# ---------------------------------------------------------------------------


def test_multiple_diag_embed_1d():
    t = torch.tensor([1., 2., 3.])
    out, dim_map = multiple_diag_embed(t, {0: 2})
    assert out.tolist() == [[1., 0., 0.], [0., 2., 0.], [0., 0., 3.]]
    # Original dim 0 expands into the two-position list [0, 1].
    assert dim_map == [[0, 1]]


def test_multiple_diag_embed_2d_one_dim():
    t = torch.arange(6.).reshape(2, 3)
    out, dim_map = multiple_diag_embed(t, {1: 2})
    # Embedding dim 1 → (k, k) diag produces shape [2, 3, 3].
    assert out.shape == (2, 3, 3)
    for i, j in itertools.product(range(2), range(3)):
        assert out[i, j, j].item() == t[i, j].item()
        assert out[i, j, (j + 1) % 3].item() == 0.
    # Dim 0 stays at [0]; dim 1 expands into two new dims [1, 2].
    assert dim_map == [[0], [1, 2]]


def test_multiple_diag_embed_two_dims_preserves_product():
    t = torch.arange(2 * 3.).reshape(2, 3)
    out, dim_map = multiple_diag_embed(t, {0: 2, 1: 2})
    assert out.shape == (2, 2, 3, 3)
    for i, j in itertools.product(range(2), range(3)):
        assert out[i, i, j, j].item() == t[i, j].item()
    # Each original dim should map to a 2-position list of tensor dims.
    assert isinstance(dim_map[0], list) and len(dim_map[0]) == 2
    assert isinstance(dim_map[1], list) and len(dim_map[1]) == 2
    # The four expanded positions are all distinct.
    positions = set(dim_map[0]) | set(dim_map[1])
    assert len(positions) == 4


def test_multiple_diagnonal_then_embed_roundtrip_values():
    """Diagonal then diag_embed should reproduce the diagonal itself."""
    t = torch.arange(4 * 4.).reshape(4, 4)
    diag, _ = multiple_diagnonal(t, [(0, 1)])
    embedded, _ = multiple_diag_embed(diag, {0: 2})
    # Embedded tensor's diagonal should equal the original diagonal.
    assert torch.equal(embedded.diagonal(), diag)


# ---------------------------------------------------------------------------
# EinsumOp.add_conditions / remove_conditions
# ---------------------------------------------------------------------------


def _full_op(shape_in, shape_out, seed=0):
    """Return a fully non-mul EinsumOp (Jacobian tensor) of the given shapes."""
    torch.manual_seed(seed)
    t = torch.randn(*shape_out, *shape_in)
    output_dims = list(range(len(shape_out)))
    input_dims = list(range(len(shape_out), len(shape_out) + len(shape_in)))
    return EinsumOp(t, input_dims, output_dims)


def _hadamard_op(shape, seed=1):
    """Return a pure hadamard EinsumOp with identical input/output shape."""
    torch.manual_seed(seed)
    t = torch.randn(*shape)
    dims = list(range(len(shape)))
    return EinsumOp(t, dims, dims)


def test_mul_conditions_basic():
    op = _full_op((3,), (5,))
    assert op.mul_conditions == EQCondition(set())
    h = _hadamard_op((4,))
    # Hadamard: input position 0 and output position 0 share a tensor dim → eqclass (0, ~0) = (0, -1).
    assert h.mul_conditions == EQCondition({(0, -1)})
    h2 = _hadamard_op((3, 4))
    assert h2.mul_conditions == EQCondition({(0, -1), (1, -2)})


def test_add_conditions_matches_forward():
    """add_conditions: adding a mul condition must not change forward semantics."""
    op = _full_op((3,), (3,), seed=10)
    assert op.mul_conditions == EQCondition(set())
    # Adding (input 0, output 0) joins them via torch.diagonal.
    fused = op.add_conditions(EQCondition({(0, -1)}))
    assert fused.mul_conditions == EQCondition({(0, -1)})
    assert fused.input_shape == op.input_shape
    assert fused.output_shape == op.output_shape
    # Forward must still match (fused is a hadamard with tensor = diag of op.tensor).
    x = torch.randn(3)
    assert torch.allclose(fused.forward(x), op.forward(x) * 0 + op.tensor.diagonal() * x) or \
           torch.allclose(fused.forward(x), op.tensor.diagonal() * x)


def test_add_conditions_rank2_forward_equivalence():
    """For a hadamard-like tensor, add_conditions should preserve forward output."""
    # Build an op with zero off-diagonal entries so the two forms truly agree.
    torch.manual_seed(20)
    diag = torch.randn(4)
    t = torch.diag(diag)  # shape [4, 4]
    op = EinsumOp(t, input_dims=[1], output_dims=[0])  # y_i = sum_j t[i,j] x_j
    assert op.mul_conditions == EQCondition(set())
    fused = op.add_conditions(EQCondition({(0, -1)}))
    assert fused.mul_conditions == EQCondition({(0, -1)})
    assert fused.tensor.shape == (4,)
    assert torch.allclose(fused.tensor, diag)
    x = torch.randn(4)
    assert torch.allclose(fused.forward(x), op.forward(x))


def test_remove_conditions_forward_equivalence():
    """remove_conditions expands via diag_embed; forward semantics must not change."""
    h = _hadamard_op((5,), seed=30)
    assert h.mul_conditions == EQCondition({(0, -1)})
    expanded = h.remove_conditions(EQCondition(set()))
    assert expanded.mul_conditions == EQCondition(set())
    assert expanded.input_shape == h.input_shape
    assert expanded.output_shape == h.output_shape
    x = torch.randn(5)
    assert torch.allclose(expanded.forward(x), h.forward(x))


def test_remove_conditions_2d_forward():
    h = _hadamard_op((3, 4), seed=31)
    assert h.mul_conditions == EQCondition({(0, -1), (1, -2)})
    expanded = h.remove_conditions(EQCondition(set()))
    assert expanded.mul_conditions == EQCondition(set())
    x = torch.randn(3, 4)
    assert torch.allclose(expanded.forward(x), h.forward(x))
    # Expanded tensor should have rank 4 and act as a block-diagonal Jacobian.
    assert expanded.tensor.dim() == 4


def test_remove_conditions_partial():
    """Remove only one of two mul conditions."""
    h = _hadamard_op((3, 4), seed=32)
    assert h.mul_conditions == EQCondition({(0, -1), (1, -2)})
    # Keep (input 1, output 1), drop (input 0, output 0).
    result = h.remove_conditions(EQCondition({(1, -2)}))
    assert result.mul_conditions == EQCondition({(1, -2)})
    x = torch.randn(3, 4)
    assert torch.allclose(result.forward(x), h.forward(x))


def test_add_then_remove_roundtrip():
    """add_conditions followed by remove_conditions should preserve forward output."""
    torch.manual_seed(40)
    diag = torch.randn(4)
    t = torch.diag(diag)
    op = EinsumOp(t, input_dims=[1], output_dims=[0])
    x = torch.randn(4)

    added = op.add_conditions(EQCondition({(0, -1)}))
    removed = added.remove_conditions(EQCondition(set()))
    assert removed.mul_conditions == EQCondition(set())
    assert torch.allclose(removed.forward(x), op.forward(x))


def test_purify_with_returns_merged_sum():
    """purify_with should return a single LinearOp equivalent to self + other."""
    h = _hadamard_op((3,), seed=60)         # mul_conditions = {(0, 0)}
    f = _full_op((3,), (3,), seed=61)       # mul_conditions = set()
    merged = h.purify_with(f)
    assert isinstance(merged, EinsumOp)
    x = torch.randn(3)
    assert torch.allclose(merged.forward(x), h.forward(x) + f.forward(x), atol=1e-5)

    # Order shouldn't matter (up to algebraic equivalence of the forward map).
    merged2 = f.purify_with(h)
    assert torch.allclose(merged2.forward(x), h.forward(x) + f.forward(x), atol=1e-5)


def test_radd_mismatched_mul_conditions():
    """Adding a hadamard and a full Jacobian EinsumOp should now succeed
    tensor-wise via purify_with rather than falling back to a generic sum."""
    torch.manual_seed(70)
    diag = torch.randn(3)
    t = torch.diag(diag)
    op_full = EinsumOp(t, input_dims=[1], output_dims=[0])  # full (no mul)
    op_had = EinsumOp(diag.clone(), input_dims=[0], output_dims=[0])  # hadamard

    combined = op_full + op_had  # exercises EinsumOp.__radd__
    assert isinstance(combined, EinsumOp)
    x = torch.randn(3)
    expected = op_full.forward(x) + op_had.forward(x)
    assert torch.allclose(combined.forward(x), expected, atol=1e-6)


# ---------------------------------------------------------------------------
# norm_input: EinsumOp, SumOp (with purify)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("p", [1, 2, 3])
def test_norm_input_einsumop_full(p):
    torch.manual_seed(100 + p)
    op = EinsumOp(torch.randn(5, 3), [1], [0])
    got = op.norm_input(p).tensor
    exp = _row_norms_from_jacobian(op.force_jacobian(), p, op.output_shape)
    assert torch.allclose(got, exp, atol=1e-5)


@pytest.mark.parametrize("p", [1, 2, 3])
def test_norm_input_einsumop_hadamard(p):
    torch.manual_seed(110 + p)
    op = EinsumOp(torch.randn(5), [0], [0])
    got = op.norm_input(p).tensor
    exp = _row_norms_from_jacobian(op.force_jacobian(), p, op.output_shape)
    assert torch.allclose(got, exp, atol=1e-5)


@pytest.mark.parametrize("p", [1, 2, 3])
def test_norm_input_einsumop_expansion(p):
    """EinsumOp with output batch dims not in input."""
    torch.manual_seed(120 + p)
    op = EinsumOp(torch.randn(4, 5, 3), [2], [0, 1])
    got = op.norm_input(p).tensor
    exp = _row_norms_from_jacobian(op.force_jacobian(), p, op.output_shape)
    assert torch.allclose(got, exp, atol=1e-5)


def _flatten_sumop_tensor(result, out_shape):
    """norm_input of a SumOp may return either a single EinsumOp (after purify
    fused all components) or a SumOp of norm results. Reduce to a tensor."""
    if isinstance(result, EinsumOp):
        return result.tensor
    if isinstance(result, SumOp):
        # Sum of per-component norms — used as an upper bound when components
        # couldn't be fused. Materialize via jacobian for a single-tensor view.
        return sum(op.tensor for op in result.ops)
    # Fall back to jacobian-based evaluation for any other LinearOp.
    jac = result.force_jacobian() if hasattr(result, "force_jacobian") else result.jacobian()
    return jac.reshape(out_shape)


@pytest.mark.parametrize("p", [1, 2, 3])
def test_norm_input_sumop_same_structure_fuses(p):
    """Two same-structure EinsumOps inside a SumOp: purify merges them into a
    single EinsumOp, so norm_input equals the row p-norm of the summed Jacobian."""
    torch.manual_seed(200 + p)
    a = EinsumOp(torch.randn(5, 3), [1], [0])
    b = EinsumOp(torch.randn(5, 3), [1], [0])
    s = SumOp(a, b)
    result = s.norm_input(p)
    got = _flatten_sumop_tensor(result, s.output_shape)
    merged_jac = a.force_jacobian() + b.force_jacobian()
    exp = _row_norms_from_jacobian(merged_jac, p, s.output_shape)
    assert torch.allclose(got, exp, atol=1e-5)
    # After purify the SumOp should have collapsed to a single op.
    assert len(s.ops) == 1 and isinstance(s.ops[0], EinsumOp)


@pytest.mark.parametrize("p", [1, 2, 3])
def test_norm_input_sumop_mixed_structure_fuses(p):
    """Hadamard + full EinsumOp: purify_with aligns conditions and fuses.
    norm_input reflects the true Jacobian of the merged operator."""
    torch.manual_seed(210 + p)
    h = EinsumOp(torch.randn(5), [0], [0])           # hadamard 5→5
    f = EinsumOp(torch.randn(5, 5), [1], [0])        # full 5→5
    s = SumOp(h, f)
    result = s.norm_input(p)
    got = _flatten_sumop_tensor(result, s.output_shape)
    merged_jac = h.force_jacobian() + f.force_jacobian()
    exp = _row_norms_from_jacobian(merged_jac, p, s.output_shape)
    assert torch.allclose(got, exp, atol=1e-5)
    assert len(s.ops) == 1 and isinstance(s.ops[0], EinsumOp)

@pytest.mark.parametrize("p", [1, 2, 3])
def test_norm_input_sumop_mixed_structure_fuses(p):
    """Full-hadamard + partial-hadamard EinsumOp on the same input/output
    shape [5, 5]: one op's ``mul_conditions`` is a strict subset of the
    other's, so ``purify_with`` aligns via ``remove_conditions`` and the sum
    collapses to a single EinsumOp. ``norm_input`` then reflects the true
    per-output row p-norm of the merged Jacobian."""
    torch.manual_seed(210 + p)
    # f: full hadamard on [5, 5] → mul_conditions = {(0, 0), (1, 1)}
    f = EinsumOp(torch.randn(3, 3, 3), [0, 1], [2, 1])
    # h: partial hadamard, mul only on dim 0 → mul_conditions = {(0, 0)}
    #    tensor[i, j, k]: y[i, k] = sum_j t[i, j, k] * x[i, j]
    h = EinsumOp(torch.randn(3, 3, 3), [0, 1], [0, 2])
    assert f.input_shape == h.input_shape == torch.Size([3, 3])
    assert f.output_shape == h.output_shape == torch.Size([3, 3])

    s = SumOp(h, f)
    jac1 = s.jacobian()
    s.purify()
    jac2 = s.jacobian()
    assert torch.allclose(jac1, jac2, atol=1e-5)
    result = s.norm_input(1)
    got = result.jacobian()
    merged_jac = h.force_jacobian() + f.force_jacobian()
    assert torch.allclose(jac1, merged_jac, atol=1e-5)
    exp = _row_norms_from_jacobian(merged_jac, 1, s.output_shape)
    assert got.shape == exp.shape == s.output_shape
    assert torch.allclose(got, exp, atol=1e-5)

@pytest.mark.parametrize("p", [1, 2, 3])
def test_norm_input_sumop_mixed_structure_fuses2(p):
    """Full-hadamard + partial-hadamard EinsumOp on the same input/output
    shape [5, 5]: one op's ``mul_conditions`` is a strict subset of the
    other's, so ``purify_with`` aligns via ``remove_conditions`` and the sum
    collapses to a single EinsumOp. ``norm_input`` then reflects the true
    per-output row p-norm of the merged Jacobian."""
    torch.manual_seed(210 + p)
    # f: full hadamard on [5, 5] → mul_conditions = {(0, 0), (1, 1)}
    f = EinsumOp(torch.randn(3, 3, 3), [0, 1], [1, 2])
    # h: partial hadamard, mul only on dim 0 → mul_conditions = {(0, 0)}
    #    tensor[i, j, k]: y[i, k] = sum_j t[i, j, k] * x[i, j]
    h = EinsumOp(torch.randn(3, 3, 3), [0, 1], [0, 2])
    assert f.input_shape == h.input_shape == torch.Size([3, 3])
    assert f.output_shape == h.output_shape == torch.Size([3, 3])

    s = SumOp(h, f)
    jac1 = s.jacobian()
    s.purify()
    jac2 = s.jacobian()
    assert torch.allclose(jac1, jac2, atol=1e-5)
    result = s.norm_input(1)
    got = result.jacobian()
    merged_jac = h.force_jacobian() + f.force_jacobian()
    assert torch.allclose(jac1, merged_jac, atol=1e-5)
    exp = _row_norms_from_jacobian(merged_jac, 1, s.output_shape)
    assert got.shape == exp.shape == s.output_shape
    assert torch.allclose(got, exp, atol=1e-5)

def test_sumop_purify_is_idempotent():
    """Calling norm_input twice shouldn't re-purify or change the result."""
    torch.manual_seed(220)
    a = EinsumOp(torch.randn(4, 3), [1], [0])
    b = EinsumOp(torch.randn(4, 3), [1], [0])
    s = SumOp(a, b)
    out1 = s.norm_input(2)
    assert s.purified
    out2 = s.norm_input(2)
    t1 = _flatten_sumop_tensor(out1, s.output_shape)
    t2 = _flatten_sumop_tensor(out2, s.output_shape)
    assert torch.allclose(t1, t2)


def test_remove_then_add_roundtrip():
    """remove_conditions followed by add_conditions must preserve forward output."""
    h = _hadamard_op((3,), seed=50)
    x = torch.randn(3)
    expanded = h.remove_conditions(EQCondition(set()))
    re_added = expanded.add_conditions(EQCondition({(0, -1)}))
    assert re_added.mul_conditions == EQCondition({(0, -1)})
    assert torch.allclose(re_added.forward(x), h.forward(x))
