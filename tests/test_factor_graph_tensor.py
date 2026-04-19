"""Tests for FactorGraphTensor low-hanging-fruit operations."""

import pytest
import torch

from boundlab.sparse.factors import FactorGraphTensor, FactorTensor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fgt(
    factors_spec: list[tuple[torch.Tensor, list[int]]],
    shape: tuple[int, ...] | list[int] | torch.Size,
) -> FactorGraphTensor:
    return FactorGraphTensor(
        [FactorTensor(t, list(d)) for t, d in factors_spec], torch.Size(shape)
    )


# ---------------------------------------------------------------------------
# Construction + to_dense + shape
# ---------------------------------------------------------------------------


class TestConstructionAndDense:
    def test_simple_outer_product(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([10.0, 20.0])
        fgt = _fgt([(a, [0]), (b, [1])], [3, 2])
        assert fgt.shape == torch.Size([3, 2])
        assert torch.equal(fgt.to_dense(), torch.outer(a, b))

    def test_three_factor_product(self):
        a = torch.tensor([1.0, 2.0])
        b = torch.tensor([3.0, 4.0, 5.0])
        c = torch.tensor([7.0, 11.0, 13.0, 17.0])
        fgt = _fgt([(a, [0]), (b, [1]), (c, [2])], [2, 3, 4])
        expected = a[:, None, None] * b[None, :, None] * c[None, None, :]
        assert torch.equal(fgt.to_dense(), expected)

    def test_shared_dim_factors(self):
        x = torch.arange(6.0).reshape(2, 3)  # dims [0, 1]
        y = torch.arange(8.0).reshape(2, 4)  # dims [0, 2]
        fgt = _fgt([(x, [0, 1]), (y, [0, 2])], [2, 3, 4])
        assert fgt.shape == torch.Size([2, 3, 4])
        expected = x[:, :, None] * y[:, None, :]
        assert torch.equal(fgt.to_dense(), expected)

    def test_subset_dims_rejected_in_constructor(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        big = torch.ones((3, 4))
        with pytest.raises(AssertionError):
            FactorGraphTensor(
                [FactorTensor(a, [0]), FactorTensor(big, [0, 1])], torch.Size([3, 4])
            )

    def test_shape_mismatch_rejected_in_constructor(self):
        # Factor's tensor size at a dim ≠ shape[dim] → immediate __post_init__ failure.
        t = torch.ones((3, 4))
        with pytest.raises(AssertionError):
            FactorGraphTensor([FactorTensor(t, [0, 1])], torch.Size([3, 5]))

    def test_uncovered_dim_allowed(self):
        # Dim 1 has no factor; to_dense broadcasts it as a constant of the declared size.
        a = torch.tensor([1.0, 2.0, 3.0])  # covers dim 0
        c = torch.tensor([10.0, 20.0])     # covers dim 2
        fgt = _fgt([(a, [0]), (c, [2])], [3, 5, 2])
        assert fgt.shape == torch.Size([3, 5, 2])
        dense = fgt.to_dense()
        assert dense.shape == torch.Size([3, 5, 2])
        # Values are constant along dim 1 (the uncovered axis).
        for j in range(5):
            assert torch.equal(dense[:, j, :], dense[:, 0, :])

    def test_all_dims_uncovered(self):
        # No factors at all — to_dense returns a ones tensor of the declared shape.
        fgt = FactorGraphTensor([], torch.Size([2, 3]))
        assert torch.equal(fgt.to_dense(), torch.ones(2, 3))

    def test_size_one_factor_dim_rejected(self):
        # Factors may not own a size-1 local axis; size-1 dims must be uncovered.
        with pytest.raises(AssertionError):
            FactorGraphTensor(
                [FactorTensor(torch.ones(1), [0])], torch.Size([1])
            )

    def test_size_one_factor_dim_in_multi_axis_rejected(self):
        # Even when the factor has other size>1 axes, a size-1 axis is disallowed.
        t = torch.ones(2, 1, 3)
        with pytest.raises(AssertionError):
            FactorGraphTensor(
                [FactorTensor(t, [0, 1, 2])], torch.Size([2, 1, 3])
            )

    def test_from_dense_roundtrip(self):
        t = torch.arange(24.0).reshape(2, 3, 4)
        fgt = FactorGraphTensor.from_dense(t)
        assert fgt.shape == t.shape
        assert len(fgt.factors) == 1
        assert fgt.factors[0].dims == [0, 1, 2]
        assert torch.equal(fgt.to_dense(), t)

    def test_from_dense_leaves_size_one_dims_uncovered(self):
        t = torch.arange(6.0).reshape(2, 1, 3)
        fgt = FactorGraphTensor.from_dense(t)
        assert fgt.shape == torch.Size([2, 1, 3])
        assert len(fgt.factors) == 1
        assert fgt.factors[0].dims == [0, 2]
        assert fgt.factors[0].tensor.shape == torch.Size([2, 3])
        assert torch.equal(fgt.to_dense(), t)

    def test_from_dense_all_size_one_dims(self):
        t = torch.tensor([[[5.0]]])  # shape (1, 1, 1)
        fgt = FactorGraphTensor.from_dense(t)
        assert fgt.shape == torch.Size([1, 1, 1])
        assert len(fgt.factors) == 1
        assert fgt.factors[0].dims == []
        assert torch.equal(fgt.to_dense(), t)

    def test_from_dense_scalar(self):
        t = torch.tensor(7.0)
        fgt = FactorGraphTensor.from_dense(t)
        assert fgt.shape == torch.Size([])
        assert torch.equal(fgt.to_dense(), t)

    def test_from_dense_handles_expanded_input(self):
        # Input has a stride-0 axis (from .expand); from_dense must materialise it.
        t = torch.tensor([1.0, 2.0, 3.0]).unsqueeze(0).expand(4, 3)
        fgt = FactorGraphTensor.from_dense(t)
        assert fgt.shape == torch.Size([4, 3])
        # After materialisation the factor axes are stride-safe.
        for f in fgt.factors:
            for d in range(f.tensor.ndim):
                assert f.tensor.stride(d) != 0
        assert torch.equal(fgt.to_dense(), t)

    def test_expanded_stride_zero_factor_dim_rejected(self):
        # A factor axis with stride 0 (created via .expand) is disallowed —
        # broadcast dims must be left uncovered.
        t = torch.tensor([1.0, 2.0]).unsqueeze(0).expand(3, 2)  # stride (0, 1)
        with pytest.raises(AssertionError):
            FactorGraphTensor(
                [FactorTensor(t, [0, 1])], torch.Size([3, 2])
            )


# ---------------------------------------------------------------------------
# __mul__ / __rmul__ / __truediv__ / __neg__
# ---------------------------------------------------------------------------


class TestMul:
    def test_scalar_mul_preserves_shape(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([10.0, 20.0])
        fgt = _fgt([(a, [0]), (b, [1])], [3, 2])
        dense = fgt.to_dense()
        out = fgt * 5.0
        assert torch.equal(out.to_dense(), dense * 5.0)
        out2 = 5.0 * fgt
        assert torch.equal(out2.to_dense(), dense * 5.0)

    def test_scalar_tensor_mul(self):
        a = torch.tensor([1.0, 2.0])
        b = torch.tensor([3.0, 4.0])
        fgt = _fgt([(a, [0]), (b, [1])], [2, 2])
        s = torch.tensor(2.5)
        assert torch.allclose((fgt * s).to_dense(), fgt.to_dense() * 2.5)

    def test_mul_two_fgts_joins_factors(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([10.0, 20.0])
        c = torch.tensor([0.5, 1.5, 2.5])
        d = torch.tensor([4.0, 5.0])
        lhs = _fgt([(a, [0]), (b, [1])], [3, 2])
        rhs = _fgt([(c, [0]), (d, [1])], [3, 2])
        out = lhs * rhs
        dim_sets = [tuple(sorted(f.dims)) for f in out.factors]
        assert sorted(dim_sets) == [(0,), (1,)]
        assert torch.allclose(out.to_dense(), lhs.to_dense() * rhs.to_dense())

    def test_mul_disjoint_dim_factors_coexist(self):
        a = torch.tensor([2.0, 3.0])
        big = torch.arange(6.0).reshape(2, 3)
        lhs = _fgt([(a, [0]), (torch.ones(3), [1])], [2, 3])
        rhs = _fgt([(big, [0, 1])], [2, 3])
        out = lhs * rhs
        # Single-dim factors absorbed into the [0,1] superset.
        assert len(out.factors) == 1
        assert set(out.factors[0].dims) == {0, 1}
        assert torch.allclose(out.to_dense(), lhs.to_dense() * rhs.to_dense())

    def test_mul_mismatched_shape_raises(self):
        a = torch.tensor([1.0, 2.0])
        b = torch.tensor([3.0, 4.0])
        lhs = _fgt([(a, [0])], [2])
        rhs = _fgt([(a, [0]), (b, [1])], [2, 2])
        with pytest.raises(AssertionError):
            _ = lhs * rhs

    def test_mul_mismatched_dim_size_raises(self):
        # Same ndim but different sizes at some dim.
        a = torch.tensor([1.0, 2.0])
        b = torch.tensor([3.0, 4.0, 5.0])
        lhs = _fgt([(a, [0])], [2])
        rhs = _fgt([(b, [0])], [3])
        with pytest.raises(AssertionError):
            _ = lhs * rhs

    def test_truediv_scalar(self):
        a = torch.tensor([2.0, 4.0, 6.0])
        b = torch.tensor([1.0, 2.0])
        fgt = _fgt([(a, [0]), (b, [1])], [3, 2])
        dense = fgt.to_dense()
        assert torch.allclose((fgt / 2.0).to_dense(), dense / 2.0)

    def test_neg(self):
        a = torch.tensor([1.0, 2.0])
        b = torch.tensor([3.0, 4.0])
        fgt = _fgt([(a, [0]), (b, [1])], [2, 2])
        assert torch.equal((-fgt).to_dense(), -fgt.to_dense())

    def test_scale_with_uncovered_dims(self):
        # Scaling an FGT whose output has uncovered dims multiplies the first
        # factor — the uncovered broadcast axes remain constant.
        a = torch.tensor([1.0, 2.0, 3.0])
        fgt = _fgt([(a, [0])], [3, 4])
        dense = fgt.to_dense()
        assert torch.allclose((fgt * 7.0).to_dense(), dense * 7.0)


# ---------------------------------------------------------------------------
# permute / transpose
# ---------------------------------------------------------------------------


class TestPermute:
    def test_permute_matches_dense(self):
        a = torch.tensor([1.0, 2.0])
        b = torch.tensor([3.0, 4.0, 5.0])
        c = torch.tensor([6.0, 7.0, 8.0, 9.0])
        fgt = _fgt([(a, [0]), (b, [1]), (c, [2])], [2, 3, 4])
        dense = fgt.to_dense()
        permuted = fgt.permute(2, 0, 1)
        assert permuted.shape == torch.Size([4, 2, 3])
        assert torch.equal(permuted.to_dense(), dense.permute(2, 0, 1))

    def test_permute_list_arg(self):
        a = torch.ones((2, 3))
        b = torch.ones((3, 4))
        fgt = _fgt([(a, [0, 1]), (b, [1, 2])], [2, 3, 4])
        dense = fgt.to_dense()
        assert torch.equal(fgt.permute([1, 2, 0]).to_dense(), dense.permute(1, 2, 0))

    def test_permute_with_uncovered_dim(self):
        a = torch.tensor([1.0, 2.0])
        fgt = _fgt([(a, [0])], [2, 5])  # dim 1 uncovered
        dense = fgt.to_dense()
        out = fgt.permute(1, 0)
        assert out.shape == torch.Size([5, 2])
        assert torch.equal(out.to_dense(), dense.permute(1, 0))

    def test_permute_invalid_raises(self):
        a = torch.tensor([1.0, 2.0])
        fgt = _fgt([(a, [0])], [2])
        with pytest.raises(AssertionError):
            fgt.permute(1)

    def test_transpose(self):
        x = torch.arange(6.0).reshape(2, 3)
        y = torch.arange(8.0).reshape(2, 4)
        fgt = _fgt([(x, [0, 1]), (y, [0, 2])], [2, 3, 4])
        dense = fgt.to_dense()
        assert torch.equal(fgt.transpose(0, 2).to_dense(), dense.transpose(0, 2))

    def test_transpose_negative_dims(self):
        a = torch.tensor([1.0, 2.0])
        b = torch.tensor([3.0, 4.0, 5.0])
        fgt = _fgt([(a, [0]), (b, [1])], [2, 3])
        dense = fgt.to_dense()
        assert torch.equal(fgt.transpose(-1, -2).to_dense(), dense.transpose(0, 1))


# ---------------------------------------------------------------------------
# squeeze / unsqueeze / expand
# ---------------------------------------------------------------------------


class TestSqueezeUnsqueeze:
    def test_unsqueeze_middle(self):
        a = torch.tensor([1.0, 2.0])
        b = torch.tensor([3.0, 4.0, 5.0])
        fgt = _fgt([(a, [0]), (b, [1])], [2, 3])
        dense = fgt.to_dense()
        out = fgt.unsqueeze(1)
        assert out.shape == torch.Size([2, 1, 3])
        assert torch.equal(out.to_dense(), dense.unsqueeze(1))

    def test_unsqueeze_is_uncovered_dim(self):
        # The newly-inserted dim has no factor covering it.
        a = torch.tensor([1.0, 2.0])
        fgt = _fgt([(a, [0])], [2])
        out = fgt.unsqueeze(1)
        covered = {d for f in out.factors for d in f.dims}
        assert 1 not in covered
        assert out.shape == torch.Size([2, 1])

    def test_unsqueeze_end(self):
        a = torch.tensor([1.0, 2.0])
        fgt = _fgt([(a, [0])], [2])
        out = fgt.unsqueeze(-1)
        assert out.shape == torch.Size([2, 1])
        assert torch.equal(out.to_dense(), a.unsqueeze(-1))

    def test_squeeze_single(self):
        # Dim 1 is size 1 and uncovered (no factor may own a size-1 dim).
        a = torch.tensor([1.0, 2.0])
        b = torch.tensor([3.0, 4.0, 5.0])
        fgt = _fgt([(a, [0]), (b, [2])], [2, 1, 3])
        out = fgt.squeeze(1)
        assert out.shape == torch.Size([2, 3])
        assert torch.equal(out.to_dense(), fgt.to_dense().squeeze(1))

    def test_squeeze_uncovered_singleton(self):
        # A size-1 dim with no factor covering it still gets squeezed away.
        a = torch.tensor([1.0, 2.0])
        fgt = _fgt([(a, [0])], [2, 1, 3])  # dim 1 uncovered and size 1, dim 2 uncovered and size 3
        out = fgt.squeeze(1)
        assert out.shape == torch.Size([2, 3])
        # Dim 2 (uncovered, size 3) shifts down to dim 1.
        assert torch.equal(out.to_dense(), fgt.to_dense().squeeze(1))

    def test_squeeze_non_singleton_is_noop(self):
        a = torch.tensor([1.0, 2.0])
        b = torch.tensor([3.0, 4.0, 5.0])
        fgt = _fgt([(a, [0]), (b, [1])], [2, 3])
        assert fgt.squeeze(0).shape == fgt.shape

    def test_unsqueeze_squeeze_roundtrip(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([4.0, 5.0])
        fgt = _fgt([(a, [0]), (b, [1])], [3, 2])
        dense = fgt.to_dense()
        out = fgt.unsqueeze(1).squeeze(1)
        assert torch.equal(out.to_dense(), dense)

    def test_squeeze_no_arg_removes_all_singleton(self):
        # Dims 1 and 2 are size 1 and uncovered.
        a = torch.tensor([1.0, 2.0])
        fgt = _fgt([(a, [0])], [2, 1, 1])
        dense = fgt.to_dense()
        out = fgt.squeeze()
        assert out.shape == torch.Size([2])
        assert torch.equal(out.to_dense(), dense.squeeze())


class TestExpand:
    def test_expand_singleton_dim(self):
        # Dim 1 is a size-1 uncovered dim; expand broadcasts it to 4.
        a = torch.tensor([1.0, 2.0])
        b = torch.tensor([3.0, 4.0, 5.0])
        fgt = _fgt([(a, [0]), (b, [2])], [2, 1, 3])
        dense = fgt.to_dense()
        out = fgt.expand(2, 4, 3)
        assert out.shape == torch.Size([2, 4, 3])
        assert torch.equal(out.to_dense(), dense.expand(2, 4, 3))

    def test_expand_uncovered_singleton(self):
        # An uncovered size-1 dim can be expanded; shape updates and to_dense broadcasts.
        a = torch.tensor([1.0, 2.0])
        fgt = _fgt([(a, [0])], [2, 1])
        dense = fgt.to_dense()
        out = fgt.expand(2, 4)
        assert out.shape == torch.Size([2, 4])
        assert torch.equal(out.to_dense(), dense.expand(2, 4))

    def test_expand_negative_one_keeps_dim(self):
        # Dim 1 is a size-1 uncovered dim; -1 keeps dims 0 and 2 as-is.
        a = torch.tensor([1.0, 2.0])
        b = torch.tensor([3.0, 4.0])
        fgt = _fgt([(a, [0]), (b, [2])], [2, 1, 2])
        dense = fgt.to_dense()
        out = fgt.expand(-1, 5, -1)
        assert out.shape == torch.Size([2, 5, 2])
        assert torch.equal(out.to_dense(), dense.expand(2, 5, 2))

    def test_expand_mismatched_non_singleton_raises(self):
        a = torch.tensor([1.0, 2.0])
        b = torch.tensor([3.0, 4.0])
        fgt = _fgt([(a, [0]), (b, [1])], [2, 2])
        with pytest.raises(AssertionError):
            fgt.expand(5, 2)

    def test_expand_list_arg(self):
        # Dim 0 is a size-1 uncovered dim; expand it to 4 via list arg.
        b = torch.tensor([3.0, 4.0])
        fgt = _fgt([(b, [1])], [1, 2])
        out = fgt.expand([4, 2])
        assert out.shape == torch.Size([4, 2])
        assert torch.equal(out.to_dense(), fgt.to_dense().expand(4, 2))


# ---------------------------------------------------------------------------
# sum / mean
# ---------------------------------------------------------------------------


class TestSum:
    def test_sum_single_dim(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([10.0, 20.0])
        fgt = _fgt([(a, [0]), (b, [1])], [3, 2])
        dense = fgt.to_dense()
        out = fgt.sum(dim=0)
        assert out.shape == torch.Size([2])
        assert torch.allclose(out.to_dense(), dense.sum(dim=0))

    def test_sum_keepdim(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([10.0, 20.0])
        fgt = _fgt([(a, [0]), (b, [1])], [3, 2])
        dense = fgt.to_dense()
        out = fgt.sum(dim=0, keepdim=True)
        assert out.shape == torch.Size([1, 2])
        assert torch.allclose(out.to_dense(), dense.sum(dim=0, keepdim=True))

    def test_sum_negative_dim(self):
        a = torch.tensor([1.0, 2.0])
        b = torch.tensor([3.0, 4.0, 5.0])
        fgt = _fgt([(a, [0]), (b, [1])], [2, 3])
        dense = fgt.to_dense()
        assert torch.allclose(fgt.sum(dim=-1).to_dense(), dense.sum(dim=-1))

    def test_sum_multiple_dims(self):
        a = torch.tensor([1.0, 2.0])
        b = torch.tensor([3.0, 4.0, 5.0])
        c = torch.tensor([6.0, 7.0])
        fgt = _fgt([(a, [0]), (b, [1]), (c, [2])], [2, 3, 2])
        dense = fgt.to_dense()
        out = fgt.sum(dim=(0, 2))
        assert out.shape == torch.Size([3])
        assert torch.allclose(out.to_dense(), dense.sum(dim=(0, 2)))

    def test_sum_all(self):
        a = torch.tensor([1.0, 2.0])
        b = torch.tensor([3.0, 4.0])
        fgt = _fgt([(a, [0]), (b, [1])], [2, 2])
        dense = fgt.to_dense()
        out = fgt.sum()
        assert out.ndim == 0
        assert torch.allclose(out.to_dense(), dense.sum())

    def test_sum_reindexes_remaining_factors(self):
        a = torch.tensor([1.0, 2.0])
        b = torch.tensor([3.0, 4.0, 5.0])
        c = torch.tensor([6.0, 7.0])
        fgt = _fgt([(a, [0]), (b, [1]), (c, [2])], [2, 3, 2])
        dense = fgt.to_dense()
        out = fgt.sum(dim=1)
        assert out.shape == torch.Size([2, 2])
        for f in out.factors:
            for d in f.dims:
                assert d < out.ndim
        assert torch.allclose(out.to_dense(), dense.sum(dim=1))

    def test_sum_over_shared_dim_fuses_via_einsum(self):
        # Two factors share dim 0. Summing it fuses them with `einsum` so the
        # dim has a single owner, then sums locally within that owner.
        x = torch.arange(6.0).reshape(2, 3)  # dims [0, 1]
        y = torch.arange(8.0).reshape(2, 4)  # dims [0, 2]
        fgt = _fgt([(x, [0, 1]), (y, [0, 2])], [2, 3, 4])
        dense = fgt.to_dense()
        out = fgt.sum(dim=0)
        assert out.shape == torch.Size([3, 4])
        assert torch.allclose(out.to_dense(), dense.sum(dim=0))

    def test_sum_keepdim_over_shared_dim(self):
        x = torch.arange(6.0).reshape(2, 3)
        y = torch.arange(8.0).reshape(2, 4)
        fgt = _fgt([(x, [0, 1]), (y, [0, 2])], [2, 3, 4])
        dense = fgt.to_dense()
        out = fgt.sum(dim=0, keepdim=True)
        assert out.shape == torch.Size([1, 3, 4])
        assert torch.allclose(out.to_dense(), dense.sum(dim=0, keepdim=True))

    def test_sum_over_shared_dim_three_factors(self):
        # Three factors share dim 0; fusion must collapse all of them.
        x = torch.arange(6.0).reshape(2, 3)     # dims [0, 1]
        y = torch.arange(8.0).reshape(2, 4)     # dims [0, 2]
        z = torch.arange(10.0).reshape(2, 5)    # dims [0, 3]
        fgt = _fgt([(x, [0, 1]), (y, [0, 2]), (z, [0, 3])], [2, 3, 4, 5])
        dense = fgt.to_dense()
        out = fgt.sum(dim=0)
        assert out.shape == torch.Size([3, 4, 5])
        assert torch.allclose(out.to_dense(), dense.sum(dim=0))

    def test_sum_over_non_shared_dim(self):
        x = torch.arange(6.0).reshape(2, 3)
        y = torch.arange(8.0).reshape(2, 4)
        fgt = _fgt([(x, [0, 1]), (y, [0, 2])], [2, 3, 4])
        dense = fgt.to_dense()
        out = fgt.sum(dim=1)
        assert out.shape == torch.Size([2, 4])
        assert torch.allclose(out.to_dense(), dense.sum(dim=1))

    def test_sum_over_uncovered_dim(self):
        # Dim 1 is uncovered (broadcast constant); summing it multiplies by size.
        a = torch.tensor([1.0, 2.0])
        fgt = _fgt([(a, [0])], [2, 5])
        dense = fgt.to_dense()
        out = fgt.sum(dim=1)
        assert out.shape == torch.Size([2])
        assert torch.allclose(out.to_dense(), dense.sum(dim=1))

    def test_sum_over_uncovered_dim_keepdim(self):
        a = torch.tensor([1.0, 2.0])
        fgt = _fgt([(a, [0])], [2, 5])
        dense = fgt.to_dense()
        out = fgt.sum(dim=1, keepdim=True)
        assert out.shape == torch.Size([2, 1])
        assert torch.allclose(out.to_dense(), dense.sum(dim=1, keepdim=True))


class TestMean:
    def test_mean_single_dim(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([10.0, 20.0])
        fgt = _fgt([(a, [0]), (b, [1])], [3, 2])
        dense = fgt.to_dense()
        assert torch.allclose(fgt.mean(dim=0).to_dense(), dense.mean(dim=0))

    def test_mean_all(self):
        a = torch.tensor([1.0, 2.0])
        b = torch.tensor([3.0, 4.0])
        fgt = _fgt([(a, [0]), (b, [1])], [2, 2])
        dense = fgt.to_dense()
        assert torch.allclose(fgt.mean().to_dense(), dense.mean())


# ---------------------------------------------------------------------------
# Canonicalisation through operation chains
# ---------------------------------------------------------------------------


class TestCanonicalisation:
    def test_mul_fuses_same_dim_set(self):
        x = torch.arange(6.0).reshape(2, 3)
        y = torch.arange(6.0).reshape(2, 3) + 1
        lhs = _fgt([(x, [0, 1])], [2, 3])
        rhs = _fgt([(y, [0, 1])], [2, 3])
        out = lhs * rhs
        assert len(out.factors) == 1
        assert set(out.factors[0].dims) == {0, 1}
        assert torch.allclose(out.to_dense(), x * y)

    def test_mul_fuses_permuted_dim_set(self):
        x = torch.arange(6.0).reshape(2, 3)
        y = torch.arange(6.0).reshape(3, 2) + 1
        lhs = _fgt([(x, [0, 1])], [2, 3])
        rhs = _fgt([(y, [1, 0])], [2, 3])
        out = lhs * rhs
        assert len(out.factors) == 1
        assert torch.allclose(out.to_dense(), x * y.t())

    def test_chain_of_operations_matches_dense(self):
        a = torch.tensor([1.0, 2.0])
        b = torch.tensor([3.0, 4.0, 5.0])
        c = torch.tensor([6.0, 7.0])
        fgt = _fgt([(a, [0]), (b, [1]), (c, [2])], [2, 3, 2])
        dense = fgt.to_dense()

        step = fgt * 2.0
        assert torch.allclose(step.to_dense(), dense * 2.0)
        step = step.permute(2, 0, 1)
        expected = (dense * 2.0).permute(2, 0, 1)
        assert torch.allclose(step.to_dense(), expected)
        step = step.sum(dim=1)
        expected = expected.sum(dim=1)
        assert torch.allclose(step.to_dense(), expected)


# ---------------------------------------------------------------------------
# Crossed 5-factor graph — every dim is shared by multiple factors
# ---------------------------------------------------------------------------
#
# Five factors over dims {0,1,2,3} with sizes [2, 3, 4, 5]:
#   t1 : [0, 1]    shape (2, 3)
#   t2 : [0, 2]    shape (2, 4)
#   t3 : [1, 3]    shape (3, 5)
#   t4 : [2, 3]    shape (4, 5)
#   t5 : [2, 1]    shape (4, 3)
# All dim-sets are distinct and none is a subset of another — canonical form.
# Every output dim is covered by >=2 factors, so summing any single dim
# exercises the einsum-fusion branch of ``_sum_one``.


SIZES = (2, 3, 4, 5)


def _build_crossed_fgt():
    # Use float64 so numerical comparisons against the dense baseline aren't
    # tripped by einsum reordering roundoff.
    torch.manual_seed(0)
    t1 = torch.randn(2, 3, dtype=torch.float64)      # dims [0, 1]
    t2 = torch.randn(2, 4, dtype=torch.float64)      # dims [0, 2]
    t3 = torch.randn(3, 5, dtype=torch.float64)      # dims [1, 3]
    t4 = torch.randn(4, 5, dtype=torch.float64)      # dims [2, 3]
    t5 = torch.randn(4, 3, dtype=torch.float64)      # dims [2, 1]
    fgt = _fgt(
        [(t1, [0, 1]), (t2, [0, 2]), (t3, [1, 3]), (t4, [2, 3]), (t5, [2, 1])],
        SIZES,
    )
    # Reference dense: product of all broadcasted factors.
    dense = (
        t1[:, :, None, None]
        * t2[:, None, :, None]
        * t3[None, :, None, :]
        * t4[None, None, :, :]
        * t5.t()[None, :, :, None]  # t5 is [2,1] → transpose to broadcast as [1, 2]
    )
    return fgt, dense


class TestCrossedFiveFactors:
    def test_construction_and_dense(self):
        fgt, dense = _build_crossed_fgt()
        assert fgt.shape == torch.Size(SIZES)
        assert len(fgt.factors) == 5
        assert torch.allclose(fgt.to_dense(), dense)

    def test_scalar_mul(self):
        fgt, dense = _build_crossed_fgt()
        assert torch.allclose((fgt * 2.5).to_dense(), dense * 2.5)
        assert torch.allclose((0.5 * fgt).to_dense(), dense * 0.5)

    def test_truediv_scalar(self):
        fgt, dense = _build_crossed_fgt()
        assert torch.allclose((fgt / 3.0).to_dense(), dense / 3.0)

    def test_neg(self):
        fgt, dense = _build_crossed_fgt()
        assert torch.allclose((-fgt).to_dense(), -dense)

    def test_mul_two_crossed_fgts(self):
        fgt, dense = _build_crossed_fgt()
        torch.manual_seed(1)
        other = _fgt(
            [
                (torch.randn(2, 3, dtype=torch.float64), [0, 1]),
                (torch.randn(3, 5, dtype=torch.float64), [1, 3]),
            ],
            SIZES,
        )
        out = fgt * other
        # ``t1`` (dims {0,1}) and other's first factor share a dim-set → fuse.
        # ``t3`` (dims {1,3}) and other's second factor share a dim-set → fuse.
        dim_sets = sorted(tuple(sorted(f.dims)) for f in out.factors)
        assert dim_sets == [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]
        assert torch.allclose(out.to_dense(), dense * other.to_dense())

    @pytest.mark.parametrize("perm", [(1, 0, 3, 2), (3, 2, 1, 0), (2, 0, 3, 1)])
    def test_permute(self, perm):
        fgt, dense = _build_crossed_fgt()
        out = fgt.permute(*perm)
        assert out.shape == torch.Size([SIZES[p] for p in perm])
        assert torch.allclose(out.to_dense(), dense.permute(*perm))

    @pytest.mark.parametrize("d0, d1", [(0, 3), (1, 2), (-1, -3)])
    def test_transpose(self, d0, d1):
        fgt, dense = _build_crossed_fgt()
        assert torch.allclose(
            fgt.transpose(d0, d1).to_dense(), dense.transpose(d0, d1)
        )

    @pytest.mark.parametrize("insert_dim", [0, 2, 4, -1])
    def test_unsqueeze(self, insert_dim):
        fgt, dense = _build_crossed_fgt()
        out = fgt.unsqueeze(insert_dim)
        assert torch.allclose(out.to_dense(), dense.unsqueeze(insert_dim))

    def test_unsqueeze_then_squeeze_roundtrip(self):
        fgt, dense = _build_crossed_fgt()
        out = fgt.unsqueeze(2).squeeze(2)
        assert out.shape == fgt.shape
        assert torch.allclose(out.to_dense(), dense)

    @pytest.mark.parametrize("insert_dim, expand_to", [(1, 7), (4, 6), (0, 3)])
    def test_expand_newly_inserted_dim(self, insert_dim, expand_to):
        # Unsqueeze inserts a size-1 uncovered dim; expand broadcasts it.
        fgt, dense = _build_crossed_fgt()
        sizes = list(fgt.shape)
        sizes.insert(insert_dim if insert_dim >= 0 else len(sizes) + 1 + insert_dim, expand_to)
        out = fgt.unsqueeze(insert_dim).expand(*sizes)
        assert out.shape == torch.Size(sizes)
        assert torch.allclose(out.to_dense(), dense.unsqueeze(insert_dim).expand(*sizes))

    @pytest.mark.parametrize("d", [0, 1, 2, 3, -1])
    def test_sum_single_dim(self, d):
        # Every output dim is shared by >=2 factors, so this exercises
        # the einsum-fusion path in ``_sum_one``.
        fgt, dense = _build_crossed_fgt()
        assert torch.allclose(fgt.sum(dim=d).to_dense(), dense.sum(dim=d))

    @pytest.mark.parametrize("d", [0, 1, 2, 3])
    def test_sum_keepdim(self, d):
        fgt, dense = _build_crossed_fgt()
        out = fgt.sum(dim=d, keepdim=True)
        expected_shape = list(SIZES)
        expected_shape[d] = 1
        assert out.shape == torch.Size(expected_shape)
        assert torch.allclose(out.to_dense(), dense.sum(dim=d, keepdim=True))

    @pytest.mark.parametrize("dims", [(0, 1), (1, 3), (0, 2, 3), (0, 1, 2, 3)])
    def test_sum_multiple_dims(self, dims):
        fgt, dense = _build_crossed_fgt()
        assert torch.allclose(fgt.sum(dim=dims).to_dense(), dense.sum(dim=dims))

    def test_sum_all(self):
        fgt, dense = _build_crossed_fgt()
        assert torch.allclose(fgt.sum().to_dense(), dense.sum())

    def test_mean_all(self):
        fgt, dense = _build_crossed_fgt()
        assert torch.allclose(fgt.mean().to_dense(), dense.mean())

    @pytest.mark.parametrize("d", [0, 1, 2, 3])
    def test_mean_single_dim(self, d):
        fgt, dense = _build_crossed_fgt()
        assert torch.allclose(fgt.mean(dim=d).to_dense(), dense.mean(dim=d))

    def test_from_dense_roundtrip_matches_factorised(self):
        fgt, dense = _build_crossed_fgt()
        round_tripped = FactorGraphTensor.from_dense(dense)
        assert torch.allclose(round_tripped.to_dense(), dense)
        assert torch.allclose(round_tripped.to_dense(), fgt.to_dense())

    def test_chained_ops_match_dense(self):
        fgt, dense = _build_crossed_fgt()
        step_fgt = (fgt * 2.0).permute(3, 0, 2, 1).transpose(1, 2).sum(dim=1)
        step_dense = (dense * 2.0).permute(3, 0, 2, 1).transpose(1, 2).sum(dim=1)
        assert torch.allclose(step_fgt.to_dense(), step_dense)


# ---------------------------------------------------------------------------
# gather_indices / scatter_indices — parity with COOMapping.forward/backward
# ---------------------------------------------------------------------------


from boundlab.sparse.factors import (
    _scatter_axis_via_indices,
    _gather_axes_via_indices,
)


def _idx(*rows):
    return torch.tensor(rows, dtype=torch.int64)


class _COOMappingStub:
    """Shim replicating the deleted COOMapping.forward/backward for parity checks."""

    def __init__(self, indices, output_shape):
        self.indices = indices
        self.output_shape = list(output_shape)

    def forward(self, t, dim):
        return _scatter_axis_via_indices(t, dim, self.indices, self.output_shape)

    def backward(self, t, dims_map):
        return _gather_axes_via_indices(t, dims_map, self.indices, self.output_shape)


COOMapping = _COOMappingStub


class TestGatherIndices:
    def test_1d_scatter_matches_coo_forward(self):
        # Single sparse dim: scatter length-2 input into length-10 output.
        t = torch.tensor([100.0, 200.0])
        fgt = FactorGraphTensor.from_dense(t)
        out = fgt.gather_indices(_idx([2, 7]), dim=0, output_shape=[10])
        expected = COOMapping(_idx([2, 7]), [10]).forward(t, dim=0)
        assert out.shape == torch.Size([10])
        assert torch.equal(out.to_dense(), expected)

    def test_2d_scatter_matches_coo_forward(self):
        # Two sparse dims: length-2 input → (4, 5) output; values only at
        # (0, 1) and (2, 3).
        t = torch.tensor([10.0, 20.0])
        fgt = FactorGraphTensor.from_dense(t)
        indices = _idx([0, 2], [1, 3])
        out = fgt.gather_indices(indices, dim=0, output_shape=[4, 5])
        expected = COOMapping(indices, [4, 5]).forward(t, dim=0)
        assert out.shape == torch.Size([4, 5])
        assert torch.equal(out.to_dense(), expected)

    def test_scatter_with_batch_dim_matches_coo_forward(self):
        # Batch dim on the left, sparse dim in the middle.
        t = torch.tensor([[10.0, 20.0], [30.0, 40.0]])  # (2, 2)
        fgt = FactorGraphTensor.from_dense(t)
        indices = _idx([1, 4])
        out = fgt.gather_indices(indices, dim=1, output_shape=[6])
        expected = COOMapping(indices, [6]).forward(t, dim=1)
        assert out.shape == torch.Size([2, 6])
        assert torch.equal(out.to_dense(), expected)

    def test_scatter_negative_dim(self):
        t = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
        fgt = FactorGraphTensor.from_dense(t)
        indices = _idx([1, 4])
        assert torch.equal(
            fgt.gather_indices(indices, dim=-1, output_shape=[6]).to_dense(),
            fgt.gather_indices(indices, dim=1, output_shape=[6]).to_dense(),
        )

    def test_scatter_on_multi_factor_fgt(self):
        # Exercise scatter on an FGT that isn't a single dense factor. Axis 0
        # is the sparse axis; it's shared between two factors.
        torch.manual_seed(42)
        t_ab = torch.randn(3, 4, dtype=torch.float64)   # dims [0, 1]
        t_ac = torch.randn(3, 5, dtype=torch.float64)   # dims [0, 2]
        fgt = _fgt([(t_ab, [0, 1]), (t_ac, [0, 2])], [3, 4, 5])
        dense = fgt.to_dense()
        indices = _idx([1, 0, 2])  # permute-style scatter of the 3 rows
        out = fgt.gather_indices(indices, dim=0, output_shape=[5])
        expected = COOMapping(indices, [5]).forward(dense, dim=0)
        assert out.shape == torch.Size([5, 4, 5])
        assert torch.allclose(out.to_dense(), expected)

    def test_scatter_shape_mismatches_raise(self):
        fgt = FactorGraphTensor.from_dense(torch.tensor([1.0, 2.0]))
        # indices column count must match shape[dim].
        with pytest.raises(AssertionError):
            fgt.gather_indices(_idx([0, 1, 2]), dim=0, output_shape=[5])
        # indices row count must match output_shape length.
        with pytest.raises(AssertionError):
            fgt.gather_indices(_idx([0, 1], [1, 0]), dim=0, output_shape=[5])


class TestScatterIndices:
    def test_1d_gather_matches_coo_backward(self):
        t = torch.arange(10, dtype=torch.float64)
        fgt = FactorGraphTensor.from_dense(t)
        indices = _idx([2, 7])
        out = fgt.scatter_indices(indices, dims_map=[0])
        expected = COOMapping(indices, [10]).backward(t, dims_map=[0])
        assert torch.equal(out.to_dense(), expected)

    def test_2d_gather_one_dim(self):
        # Pick 2 columns out of a (3, 5) tensor.
        t = torch.arange(15, dtype=torch.float64).reshape(3, 5)
        fgt = FactorGraphTensor.from_dense(t)
        indices = _idx([1, 3])
        out = fgt.scatter_indices(indices, dims_map=[1])
        expected = COOMapping(indices, [5]).backward(t, dims_map=[1])
        assert out.shape == torch.Size([3, 2])
        assert torch.equal(out.to_dense(), expected)

    def test_gather_contiguous_dims_lands_at_min(self):
        # Two mapped dims [1, 2]: contiguous → collapsed axis at position 1.
        t = torch.arange(2 * 4 * 5, dtype=torch.float64).reshape(2, 4, 5)
        fgt = FactorGraphTensor.from_dense(t)
        indices = _idx([0, 2], [1, 4])  # pick (0,1), (2,4) across dims 1 & 2
        out = fgt.scatter_indices(indices, dims_map=[1, 2])
        expected = COOMapping(indices, [4, 5]).backward(t, dims_map=[1, 2])
        assert out.shape == expected.shape  # (2, 2)
        assert torch.equal(out.to_dense(), expected)

    def test_gather_noncontiguous_dims_lands_at_zero(self):
        # Mapped dims [0, 2]: noncontiguous → collapsed axis at position 0.
        t = torch.arange(3 * 4 * 5, dtype=torch.float64).reshape(3, 4, 5)
        fgt = FactorGraphTensor.from_dense(t)
        indices = _idx([0, 2], [1, 4])
        out = fgt.scatter_indices(indices, dims_map=[0, 2])
        expected = COOMapping(indices, [3, 5]).backward(t, dims_map=[0, 2])
        assert out.shape == expected.shape  # (2, 4) — K at front, dim 1 kept
        assert torch.equal(out.to_dense(), expected)

    def test_gather_negative_dim(self):
        t = torch.arange(15, dtype=torch.float64).reshape(3, 5)
        fgt = FactorGraphTensor.from_dense(t)
        indices = _idx([1, 3])
        assert torch.equal(
            fgt.scatter_indices(indices, dims_map=[-1]).to_dense(),
            fgt.scatter_indices(indices, dims_map=[1]).to_dense(),
        )

    def test_gather_on_multi_factor_fgt(self):
        torch.manual_seed(7)
        t_ab = torch.randn(3, 4, dtype=torch.float64)
        t_ac = torch.randn(3, 5, dtype=torch.float64)
        fgt = _fgt([(t_ab, [0, 1]), (t_ac, [0, 2])], [3, 4, 5])
        dense = fgt.to_dense()
        indices = _idx([0, 2], [1, 4])  # index across dims 1 & 2 (contiguous)
        out = fgt.scatter_indices(indices, dims_map=[1, 2])
        expected = COOMapping(indices, [4, 5]).backward(dense, dims_map=[1, 2])
        assert torch.allclose(out.to_dense(), expected)

    def test_gather_dims_map_mismatches_indices_raises(self):
        fgt = FactorGraphTensor.from_dense(torch.arange(20, dtype=torch.float64).reshape(4, 5))
        # 2 rows of indices but only 1 entry in dims_map.
        with pytest.raises(AssertionError):
            fgt.scatter_indices(_idx([1, 3], [0, 2]), dims_map=[0])


class TestGatherScatterRoundtrip:
    def test_scatter_then_gather_recovers_input(self):
        # COOMapping.backward ∘ COOMapping.forward = identity on the sparse
        # positions (which are exactly what the input occupies).
        torch.manual_seed(3)
        t = torch.randn(2, dtype=torch.float64)
        fgt = FactorGraphTensor.from_dense(t)
        indices = _idx([0, 2], [1, 3])  # 2 sparse dims, K=2 entries
        scattered = fgt.gather_indices(indices, dim=0, output_shape=[3, 5])
        gathered = scattered.scatter_indices(indices, dims_map=[0, 1])
        assert gathered.shape == torch.Size([2])
        assert torch.allclose(gathered.to_dense(), t)


class TestAdd:
    def test_add_same_dim_set(self):
        torch.manual_seed(0)
        a = torch.randn(3, dtype=torch.float64)
        b = torch.randn(3, dtype=torch.float64)
        lhs = _fgt([(a, [0])], [3])
        rhs = _fgt([(b, [0])], [3])
        out = lhs + rhs
        assert [sorted(f.dims) for f in out.factors] == [[0]]
        assert torch.allclose(out.to_dense(), lhs.to_dense() + rhs.to_dense())

    def test_add_disjoint_dim_sets_unions(self):
        torch.manual_seed(1)
        a = torch.randn(3, dtype=torch.float64)
        b = torch.randn(4, dtype=torch.float64)
        lhs = _fgt([(a, [0])], [3, 4])
        rhs = _fgt([(b, [1])], [3, 4])
        out = lhs + rhs
        assert len(out.factors) == 1
        assert sorted(out.factors[0].dims) == [0, 1]
        assert torch.allclose(out.to_dense(), lhs.to_dense() + rhs.to_dense())

    def test_add_overlapping_dim_sets_unions(self):
        torch.manual_seed(2)
        t_ab = torch.randn(3, 4, dtype=torch.float64)
        t_bc = torch.randn(4, 5, dtype=torch.float64)
        lhs = _fgt([(t_ab, [0, 1])], [3, 4, 5])
        rhs = _fgt([(t_bc, [1, 2])], [3, 4, 5])
        out = lhs + rhs
        assert len(out.factors) == 1
        assert sorted(out.factors[0].dims) == [0, 1, 2]
        assert torch.allclose(out.to_dense(), lhs.to_dense() + rhs.to_dense())

    def test_add_subset_dim_sets_unions(self):
        torch.manual_seed(3)
        a = torch.randn(3, dtype=torch.float64)
        t_ab = torch.randn(3, 4, dtype=torch.float64)
        lhs = _fgt([(a, [0])], [3, 4])
        rhs = _fgt([(t_ab, [0, 1])], [3, 4])
        out = lhs + rhs
        assert len(out.factors) == 1
        assert sorted(out.factors[0].dims) == [0, 1]
        assert torch.allclose(out.to_dense(), lhs.to_dense() + rhs.to_dense())

    def test_add_scalar_int(self):
        torch.manual_seed(4)
        a = torch.randn(3, dtype=torch.float64)
        b = torch.randn(4, dtype=torch.float64)
        fgt = _fgt([(a, [0]), (b, [1])], [3, 4])
        out = fgt + 3
        assert torch.allclose(out.to_dense(), fgt.to_dense() + 3)

    def test_add_scalar_float(self):
        torch.manual_seed(5)
        a = torch.randn(2, 3, dtype=torch.float64)
        fgt = _fgt([(a, [0, 1])], [2, 3])
        out = fgt + 1.5
        assert torch.allclose(out.to_dense(), fgt.to_dense() + 1.5)

    def test_radd_scalar(self):
        torch.manual_seed(6)
        a = torch.randn(3, dtype=torch.float64)
        fgt = _fgt([(a, [0])], [3])
        out = 7.0 + fgt
        assert torch.allclose(out.to_dense(), 7.0 + fgt.to_dense())

    def test_add_zero_dim_tensor(self):
        torch.manual_seed(7)
        a = torch.randn(3, 4, dtype=torch.float64)
        fgt = _fgt([(a, [0, 1])], [3, 4])
        s = torch.tensor(2.5, dtype=torch.float64)
        out = fgt + s
        assert torch.allclose(out.to_dense(), fgt.to_dense() + 2.5)

    def test_add_non_scalar_tensor_returns_notimplemented(self):
        a = torch.tensor([1.0, 2.0])
        fgt = _fgt([(a, [0])], [2])
        with pytest.raises(TypeError):
            _ = fgt + torch.tensor([1.0, 2.0])

    def test_add_shape_mismatch_raises(self):
        a = torch.tensor([1.0, 2.0])
        b = torch.tensor([3.0, 4.0, 5.0])
        lhs = _fgt([(a, [0])], [2])
        rhs = _fgt([(b, [0])], [3])
        with pytest.raises(AssertionError):
            _ = lhs + rhs

    def test_add_preserves_uncovered_dims(self):
        # Uncovered dim 2 (size 5) on both sides: the union only covers {0, 1},
        # so dim 2 stays uncovered in the result and broadcasts via to_dense.
        torch.manual_seed(8)
        a = torch.randn(3, dtype=torch.float64)
        b = torch.randn(4, dtype=torch.float64)
        lhs = _fgt([(a, [0])], [3, 4, 5])
        rhs = _fgt([(b, [1])], [3, 4, 5])
        out = lhs + rhs
        assert out.shape == torch.Size([3, 4, 5])
        assert sorted(out.factors[0].dims) == [0, 1]
        assert torch.allclose(out.to_dense(), lhs.to_dense() + rhs.to_dense())

    def test_add_empty_factors_both_sides(self):
        lhs = FactorGraphTensor([], torch.Size([3, 4]))
        rhs = FactorGraphTensor([], torch.Size([3, 4]))
        out = lhs + rhs
        # Both sides fuse to scalar 1.0 → result is scalar 2.0 with no covered dims.
        assert len(out.factors) == 1
        assert out.factors[0].dims == []
        assert torch.allclose(out.to_dense(), torch.full((3, 4), 2.0))

    def test_add_empty_factors_one_side(self):
        torch.manual_seed(9)
        a = torch.randn(3, dtype=torch.float64)
        lhs = FactorGraphTensor([], torch.Size([3]))
        rhs = _fgt([(a, [0])], [3])
        out = lhs + rhs
        assert sorted(out.factors[0].dims) == [0]
        assert torch.allclose(out.to_dense(), lhs.to_dense() + rhs.to_dense())

    def test_add_multi_factor_fgts(self):
        torch.manual_seed(10)
        a = torch.randn(3, dtype=torch.float64)
        b = torch.randn(4, dtype=torch.float64)
        c = torch.randn(3, 4, dtype=torch.float64)
        d = torch.randn(4, 5, dtype=torch.float64)
        lhs = _fgt([(a, [0]), (b, [1])], [3, 4, 5])
        rhs = _fgt([(c, [0, 1]), (d, [1, 2])], [3, 4, 5])
        out = lhs + rhs
        assert sorted(out.factors[0].dims) == [0, 1, 2]
        assert torch.allclose(out.to_dense(), lhs.to_dense() + rhs.to_dense())

    def test_add_negation_yields_zero(self):
        torch.manual_seed(11)
        a = torch.randn(3, dtype=torch.float64)
        b = torch.randn(4, dtype=torch.float64)
        fgt = _fgt([(a, [0]), (b, [1])], [3, 4])
        out = fgt + (-fgt)
        assert torch.allclose(out.to_dense(), torch.zeros(3, 4, dtype=torch.float64))

    def test_add_result_satisfies_invariants(self):
        # The result factor must have size>1 and stride!=0 at every covered dim.
        torch.manual_seed(12)
        a = torch.randn(3, dtype=torch.float64)
        b = torch.randn(4, dtype=torch.float64)
        lhs = _fgt([(a, [0])], [3, 4])
        rhs = _fgt([(b, [1])], [3, 4])
        out = lhs + rhs
        f = out.factors[0]
        for i, d in enumerate(f.dims):
            assert f.tensor.shape[i] > 1
            assert f.tensor.stride(i) != 0
