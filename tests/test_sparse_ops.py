import pytest
import torch
from boundlab.sparse.ops import table_join_sorted
from boundlab.sparse.factors import (
    FactorGraphTensor,
    FactorTensor,
    _gather_axes_via_indices,
    _scatter_axis_via_indices,
)
from boundlab.sparse.coos import (
    COOCondition,
    COOConditions,
    MultiCOOTensor,
    MultiCOOTensorSum,
)
from boundlab.sparse.table import TorchTable


def t32(*rows):
    return torch.tensor(rows, dtype=torch.int32)


def idx(*rows):
    return torch.tensor(rows, dtype=torch.int64)


# ---------------------------------------------------------------------------
# Test helpers — build COOCondition / forward / backward without COOMapping.
# ---------------------------------------------------------------------------


def _cond(dim_ids, rows, output_shape):
    """Build a COOCondition from dim labels + per-dim row values.

    ``rows[k]`` is the sequence of values (length N) for column ``dim_ids[k]``.
    """
    data = [
        r.to(torch.int64) if isinstance(r, torch.Tensor) else torch.tensor(r, dtype=torch.int64)
        for r in rows
    ]
    return COOCondition(TorchTable(list(dim_ids), data), list(output_shape))


def _cond_from_indices(indices, dim_ids, output_shape):
    """Build a COOCondition from a (K, N) indices tensor and matching dim labels."""
    data = [indices[k].to(torch.int64) for k in range(indices.shape[0])]
    return COOCondition(TorchTable(list(dim_ids), data), list(output_shape))


def _eye_cond(dim_ids, length):
    """COOCondition whose table is an all-indices MDEye-style block of ``length`` rows."""
    k = len(dim_ids)
    table = TorchTable(list(dim_ids), [None] * k, length=length)
    table.is_sorted = True
    table.is_unique = True
    return COOCondition(table, [length] * k)


def _scatter(tensor, dim, indices, output_shape):
    return _scatter_axis_via_indices(tensor, dim, indices.to(torch.int64), list(output_shape))


def _gather(tensor, dims_map, indices, output_shape):
    return _gather_axes_via_indices(tensor, list(dims_map), indices.to(torch.int64), list(output_shape))


def _rows_of(cond: COOCondition) -> set:
    """Set of row-tuples in a COOCondition's table."""
    mat = cond.table.materialize()
    return {tuple(mat[i].tolist()) for i in range(mat.shape[0])}


# ---------------------------------------------------------------------------
# table_join_sorted
# ---------------------------------------------------------------------------


def test_basic_two_table_join():
    A = t32([1, 10], [2, 20], [3, 30])
    B = t32([1, 100], [2, 200])
    result = table_join_sorted(A, [0, 1], B, [0, 2])
    assert result.shape == (2, 3)
    rows = {row[0].item(): row for row in result}
    assert rows[1][1].item() == 10
    assert rows[1][2].item() == 100
    assert rows[2][1].item() == 20
    assert rows[2][2].item() == 200


def test_no_matching_rows():
    A = t32([1, 10], [2, 20])
    B = t32([3, 300], [4, 400])
    result = table_join_sorted(A, [0, 1], B, [0, 2])
    assert result.shape[0] == 0


def test_single_table_identity():
    A = t32([1, 10], [2, 20], [3, 30])
    result = table_join_sorted(A, [0, 1])
    assert result.shape == (3, 2)
    assert torch.equal(result, A)


def test_three_table_join():
    A = t32([1, 10], [2, 20])
    B = t32([1, 100], [2, 200])
    C = t32([1, 1000], [2, 2000])
    result = table_join_sorted(A, [0, 1], B, [0, 2], C, [0, 3])
    assert result.shape == (2, 4)
    rows = {row[0].item(): row for row in result}
    assert rows[1][1].item() == 10
    assert rows[1][2].item() == 100
    assert rows[1][3].item() == 1000


def test_dtype_int64():
    A = torch.tensor([[1, 10], [2, 20]], dtype=torch.int64)
    B = torch.tensor([[1, 100]], dtype=torch.int64)
    result = table_join_sorted(A, [0, 1], B, [0, 2])
    assert result.shape == (1, 3)


def test_wrong_dtype_raises():
    A = torch.tensor([[1.0, 10.0], [2.0, 20.0]])
    B = torch.tensor([[1, 100]], dtype=torch.int32)
    with pytest.raises((AssertionError, Exception)):
        table_join_sorted(A, [0, 1], B, [0, 2])


# ---------------------------------------------------------------------------
# list_index_unique
# ---------------------------------------------------------------------------

from boundlab.sparse.ops import list_index_unique


def test_list_index_unique_basic():
    tensor1 = torch.tensor([[1, 2], [3, 4], [5, 6]])
    tensor2 = torch.tensor([[3, 4], [1, 2]])
    out = list_index_unique(tensor1, tensor2)
    assert out.tolist() == [1, 0]


def test_list_index_unique_single_row():
    tensor1 = torch.tensor([[10, 20], [30, 40]])
    tensor2 = torch.tensor([[30, 40]])
    out = list_index_unique(tensor1, tensor2)
    assert out.tolist() == [1]


def test_list_index_unique_identity_order():
    tensor1 = torch.tensor([[1], [2], [3], [4]])
    tensor2 = torch.tensor([[1], [2], [3], [4]])
    out = list_index_unique(tensor1, tensor2)
    assert out.tolist() == [0, 1, 2, 3]


def test_list_index_unique_reverse_order():
    tensor1 = torch.tensor([[1], [2], [3]])
    tensor2 = torch.tensor([[3], [2], [1]])
    out = list_index_unique(tensor1, tensor2)
    assert out.tolist() == [2, 1, 0]


def test_list_index_unique_multi_dim():
    tensor1 = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    tensor2 = torch.tensor([[[5, 6], [7, 8]], [[1, 2], [3, 4]]])
    out = list_index_unique(tensor1, tensor2)
    assert out.tolist() == [1, 0]


# ---------------------------------------------------------------------------
# MultiCOOTensor / MultiCOOTensorSum
# ---------------------------------------------------------------------------


def _mct(tensor: torch.Tensor, conds: COOConditions) -> MultiCOOTensor:
    return MultiCOOTensor(FactorGraphTensor.from_dense(tensor), conds)


def _single_term_sum(tensor: torch.Tensor) -> MultiCOOTensorSum:
    conds = COOConditions([_eye_cond([0], length=tensor.shape[0])])
    return MultiCOOTensorSum([_mct(tensor, conds)])


def _dense_of(s: MultiCOOTensorSum) -> torch.Tensor:
    assert len(s.terms) > 0
    parts = [t.conditions.forward(t.tensor.to_dense()) for t in s.terms]
    out = parts[0].clone()
    for p in parts[1:]:
        out = out + p
    return out


def _mct_dense(m: MultiCOOTensor) -> torch.Tensor:
    return m.conditions.forward(m.tensor.to_dense())


def _permuted_conds():
    """K=2 + K=1 conditions with interleaved output dims (permutation [0, 2, 1])."""
    c_big = _cond([0, 2], [[0, 2], [3, 5]], [4, 6])  # entries: (0,3), (2,5)
    c_small = _cond([1], [[1, 3]], [5])               # entries: 1, 3
    return COOConditions([c_big, c_small])


def _triple_permuted_conds():
    """Single K=3 condition whose dims [2, 0, 1] force a non-identity permutation."""
    c = _cond([2, 0, 1], [[2, 7], [1, 3], [0, 4]], [8, 5, 6])
    return COOConditions([c])


# --- apply_multiplicative -------------------------------------------------


class TestApplyMultiplicative:
    def test_applies_fn_to_single_term(self):
        t = torch.arange(4, dtype=torch.int64)
        s = _single_term_sum(t)
        result = s.apply_multiplicative(lambda x: x * 2)
        assert torch.equal(_dense_of(result), t * 2)

    def test_identity_fn_matches_sum(self):
        t = torch.tensor([7, 11, 13, 17], dtype=torch.int64)
        s = _single_term_sum(t)
        result = s.apply_multiplicative(lambda x: x)
        assert torch.equal(_dense_of(result), t)

    def test_nonlinear_fn(self):
        t = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64)
        s = _single_term_sum(t)
        result = s.apply_multiplicative(lambda x: x * x)
        assert torch.equal(_dense_of(result), t * t)

    def test_returns_multi_coo_tensor_sum(self):
        t = torch.arange(3, dtype=torch.int64)
        s = _single_term_sum(t)
        result = s.apply_multiplicative(lambda x: x)
        assert isinstance(result, MultiCOOTensorSum)
        assert len(result.terms) == 1
        assert result.terms[0].conditions is s.terms[0].conditions

    def test_fn_is_invoked_once_per_term(self):
        calls = []

        def fn(x):
            calls.append(x)
            return x + 1

        t = torch.arange(3, dtype=torch.int64)
        s = _single_term_sum(t)
        result = s.apply_multiplicative(fn)
        assert len(calls) == 1
        assert torch.equal(calls[0], t)
        assert torch.equal(_dense_of(result), t + 1)


# --- Non-trivial MultiCOOTensor -------------------------------------------


class TestNonTrivialMultiCOOTensor:
    def test_triple_permuted_forward_and_backward(self):
        conds = _triple_permuted_conds()
        assert conds.dim() == 3
        assert conds.permutation_forward() == [2, 0, 1]

        t = torch.tensor([100, 200], dtype=torch.int64)
        out = conds.forward(t)
        assert out.shape == torch.Size([5, 6, 8])
        assert out[1, 0, 2].item() == 100
        assert out[3, 4, 7].item() == 200
        assert (out != 0).sum().item() == 2

        back = conds.backward(out)
        assert torch.equal(back, t)

    def test_mixed_k_forward_and_backward(self):
        conds = _permuted_conds()
        assert conds.dim() == 3
        assert conds.permutation_forward() == [0, 2, 1]

        t = torch.tensor([[11, 22], [33, 44]], dtype=torch.int64)
        out = conds.forward(t)
        assert out.shape == torch.Size([4, 5, 6])

        back = conds.backward(out)
        assert torch.equal(back, t)

    def test_apply_multiplicative_preserves_conditions(self):
        conds = _permuted_conds()
        t = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)
        mct = _mct(t, conds)
        squared = mct.apply_multiplicative(lambda x: x * x)
        assert squared.conditions is conds
        assert torch.equal(squared.tensor.to_dense(), t * t)

    def test_apply_multiplicative_inplace_mutates_tensor(self):
        conds = _permuted_conds()
        t = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)
        mct = _mct(t, conds)
        mct.apply_multiplicative_(lambda x: x + 10)
        assert torch.equal(mct.tensor.to_dense(), torch.tensor([[11, 12], [13, 14]]))
        assert mct.conditions is conds

    def test_add_same_conditions(self):
        conds = _triple_permuted_conds()
        t1 = torch.tensor([10, 20], dtype=torch.int64)
        t2 = torch.tensor([1, 2], dtype=torch.int64)
        merged = _mct(t1, conds) + _mct(t2, conds)
        assert merged is not None
        assert merged.conditions is conds
        assert torch.equal(merged.tensor.to_dense(), t1 + t2)

    def test_sub_same_conditions(self):
        conds = _permuted_conds()
        t1 = torch.tensor([[5, 6], [7, 8]], dtype=torch.int64)
        t2 = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)
        diff = _mct(t1, conds) - _mct(t2, conds)
        assert diff is not None
        assert torch.equal(diff.tensor.to_dense(), t1 - t2)


# --- MultiCOOTensorSum overlap --------------------------------------------


class TestMultiCOOTensorSumOverlap:
    def test_add_term_merges_identical_conditions(self):
        conds = _triple_permuted_conds()
        t1 = torch.tensor([10, 20], dtype=torch.int64)
        t2 = torch.tensor([1, 2], dtype=torch.int64)
        s = MultiCOOTensorSum([])
        s.add_term(_mct(t1, conds))
        assert len(s.terms) == 1
        s.add_term(_mct(t2, conds))
        assert len(s.terms) == 1
        assert torch.equal(s.terms[0].tensor.to_dense(), t1 + t2)

    def test_add_term_keeps_disjoint_conditions_separate(self):
        conds_a = _triple_permuted_conds()
        conds_b = _permuted_conds()
        t_a = torch.tensor([1, 2], dtype=torch.int64)
        t_b = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)
        s = MultiCOOTensorSum([])
        s.add_term(_mct(t_a, conds_a))
        s.add_term(_mct(t_b, conds_b))
        assert len(s.terms) == 2

    def test_apply_multiplicative_after_overlap_fusion(self):
        conds = _permuted_conds()
        tensors = [
            torch.tensor([[1, 1], [1, 1]], dtype=torch.int64),
            torch.tensor([[2, 3], [4, 5]], dtype=torch.int64),
            torch.tensor([[10, 20], [30, 40]], dtype=torch.int64),
        ]
        s = MultiCOOTensorSum([])
        for t in tensors:
            s.add_term(_mct(t, conds))
        assert len(s.terms) == 1
        expected_sum = sum(tensors[1:], tensors[0])
        result = s.apply_multiplicative(lambda x: x * x)
        expected = conds.forward(expected_sum) ** 2
        assert torch.equal(_dense_of(result), expected)

    def test_apply_multiplicative_non_trivial_single_term(self):
        conds = _triple_permuted_conds()
        t = torch.tensor([3, 4], dtype=torch.int64)
        s = MultiCOOTensorSum([_mct(t, conds)])
        result = s.apply_multiplicative(lambda x: x * x)
        assert torch.equal(_dense_of(result), conds.forward(t) ** 2)

    def test_forward_then_apply_matches_apply_multiplicative(self):
        conds = _permuted_conds()
        t = torch.tensor([[2, 3], [5, 7]], dtype=torch.int64)
        full = conds.forward(t)
        direct = full * full
        s = MultiCOOTensorSum([_mct(t, conds)])
        result = s.apply_multiplicative(lambda x: x * x)
        assert torch.equal(_dense_of(result), direct)


# --- Non-mergeable MCTs ---------------------------------------------------


class TestNonMergeableMultiCOOTensorSum:
    """Rank-4 dense space with incompatible couplings → add_term keeps them
    as two separate terms, apply_multiplicative sums them densely."""

    def _build(self):
        D0, D1, D2, D3 = 4, 5, 6, 7
        conds1 = COOConditions([
            _cond([0, 2], [[0, 1], [2, 4]], [D0, D2]),  # (0,2), (1,4)
            _cond([1], [[1, 3]], [D1]),
            _cond([3], [[0, 5]], [D3]),
        ])
        conds2 = COOConditions([
            _cond([0, 1], [[0, 1, 2], [1, 3, 0]], [D0, D1]),  # (0,1),(1,3),(2,0)
            _cond([2, 3], [[2, 4], [0, 5]], [D2, D3]),
        ])
        t1 = torch.ones((2, 2, 2), dtype=torch.int64)
        t2 = torch.ones((3, 2), dtype=torch.int64)
        return conds1, conds2, t1, t2

    def test_neither_condition_set_is_subset(self):
        conds1, conds2, _, _ = self._build()
        assert not conds1.is_subset(conds2)
        assert not conds2.is_subset(conds1)

    def test_add_returns_none_for_distinct_couplings(self):
        conds1, conds2, t1, t2 = self._build()
        mct1 = _mct(t1, conds1)
        mct2 = _mct(t2, conds2)
        assert (mct1 + mct2) is None
        assert (mct2 + mct1) is None

    def test_add_term_keeps_distinct_couplings_separate(self):
        conds1, conds2, t1, t2 = self._build()
        s = MultiCOOTensorSum([])
        s.add_term(_mct(t1, conds1))
        s.add_term(_mct(t2, conds2))
        assert len(s.terms) == 2
        assert s.terms[0].conditions is conds1
        assert s.terms[1].conditions is conds2

    def test_apply_multiplicative_with_non_empty_overlap(self):
        conds1, conds2, t1, t2 = self._build()
        full1 = conds1.forward(t1)
        full2 = conds2.forward(t2)
        overlap = (full1 != 0) & (full2 != 0)
        assert overlap.sum().item() == 2
        assert overlap[0, 1, 2, 0] and overlap[1, 3, 4, 5]

        s = MultiCOOTensorSum([])
        s.add_term(_mct(t1, conds1))
        s.add_term(_mct(t2, conds2))
        assert len(s.terms) == 2

        result = s.apply_multiplicative(lambda x: x * x)
        dense_result = _dense_of(result)
        expected = (full1 + full2) ** 2
        assert torch.equal(dense_result, expected)
        assert dense_result[0, 1, 2, 0].item() == 4
        assert dense_result[1, 3, 4, 5].item() == 4


# --- from_dense -----------------------------------------------------------


class TestMultiCOOTensorFromDense:
    def _dense_with_coupled_support(self):
        dense = torch.zeros((4, 5, 6), dtype=torch.int64)
        dense[0, 1, 3] = 11
        dense[0, 3, 3] = 22
        dense[2, 1, 5] = 33
        dense[2, 3, 5] = 44
        return dense

    def test_builds_expected_conditions(self):
        dense = self._dense_with_coupled_support()
        mct = MultiCOOTensor.from_dense(dense, [[0, 2], [1]])
        assert mct.conditions.dim() == 3
        c_big, c_small = mct.conditions[0], mct.conditions[1]
        assert c_big.dims == [0, 2] and c_small.dims == [1]
        assert _rows_of(c_big) == {(0, 3), (2, 5)}
        assert _rows_of(c_small) == {(1,), (3,)}

    def test_forward_reconstructs_dense_input(self):
        dense = self._dense_with_coupled_support()
        mct = MultiCOOTensor.from_dense(dense, [[0, 2], [1]])
        assert torch.equal(mct.conditions.forward(mct.tensor.to_dense()), dense)

    def test_roundtrip_over_permuted_conds(self):
        conds = _permuted_conds()
        t = torch.tensor([[11, 22], [33, 44]], dtype=torch.int64)
        full = conds.forward(t)
        mct = MultiCOOTensor.from_dense(full, [[0, 2], [1]])
        assert torch.equal(mct.conditions.forward(mct.tensor.to_dense()), full)

    def test_zero_tensor_yields_empty_mappings(self):
        dense = torch.zeros((4, 5, 6), dtype=torch.int64)
        mct = MultiCOOTensor.from_dense(dense, [[0, 2], [1]])
        for c in mct.conditions:
            assert c.is_empty()
        assert mct.tensor.numel() == 0

    def test_out_of_support_values_are_discarded(self):
        dense = torch.zeros((4, 5, 6), dtype=torch.int64)
        dense[0, 1, 3] = 7
        dense[2, 3, 5] = 9
        mct = MultiCOOTensor.from_dense(dense, [[0, 2], [1]])
        reconstructed = mct.conditions.forward(mct.tensor.to_dense())
        assert torch.equal(reconstructed[dense != 0], dense[dense != 0])
        assert reconstructed[0, 3, 3].item() == 0
        assert reconstructed[2, 1, 5].item() == 0

    def test_rejects_non_partition(self):
        dense = torch.zeros((3, 3), dtype=torch.int64)
        with pytest.raises(AssertionError):
            MultiCOOTensor.from_dense(dense, [[0]])
        with pytest.raises(AssertionError):
            MultiCOOTensor.from_dense(dense, [[0], [0, 1]])


# --- MultiCOOTensorSum + from_dense ---------------------------------------


class TestMultiCOOTensorSumFromDense:
    def _pair_on_shared_conditions(self):
        dense_a = torch.zeros((4, 5, 6), dtype=torch.int64)
        dense_a[0, 1, 3] = 5
        dense_a[2, 3, 5] = 9
        mct_a = MultiCOOTensor.from_dense(dense_a, [[0, 2], [1]])
        dense_b = torch.zeros((4, 5, 6), dtype=torch.int64)
        dense_b[0, 1, 3] = 1
        dense_b[2, 3, 5] = 2
        mct_b = MultiCOOTensor(
            FactorGraphTensor.from_dense(mct_a.conditions.backward(dense_b)),
            mct_a.conditions,
        )
        return mct_a, mct_b, dense_a, dense_b

    def test_shared_conditions_fuse_in_add_term(self):
        mct_a, mct_b, _, _ = self._pair_on_shared_conditions()
        s = MultiCOOTensorSum([])
        s.add_term(mct_a)
        s.add_term(mct_b)
        assert len(s.terms) == 1
        assert torch.equal(
            s.terms[0].tensor.to_dense(),
            mct_a.tensor.to_dense() + mct_b.tensor.to_dense(),
        )

    def test_shared_conditions_apply_multiplicative_matches_dense_sum(self):
        mct_a, mct_b, dense_a, dense_b = self._pair_on_shared_conditions()
        s = MultiCOOTensorSum([mct_a, mct_b])
        result = s.apply_multiplicative(lambda x: x * x)
        assert torch.equal(_dense_of(result), (dense_a + dense_b) ** 2)

    def test_disjoint_couplings_stay_separate(self):
        shape = (4, 5, 6)
        dense_a = torch.zeros(shape, dtype=torch.int64)
        dense_a[0, 1, 3] = 7
        dense_a[2, 3, 5] = 11
        dense_b = torch.zeros(shape, dtype=torch.int64)
        dense_b[0, 1, 3] = 2
        dense_b[3, 2, 4] = 6

        mct_a = MultiCOOTensor.from_dense(dense_a, [[0, 2], [1]])
        mct_b = MultiCOOTensor.from_dense(dense_b, [[0, 1], [2]])

        assert not mct_a.conditions.is_subset(mct_b.conditions)
        assert not mct_b.conditions.is_subset(mct_a.conditions)

        s = MultiCOOTensorSum([])
        s.add_term(mct_a)
        s.add_term(mct_b)
        assert len(s.terms) == 2

        result = s.apply_multiplicative(lambda x: x * x)
        dense_result = _dense_of(result)
        expected = (dense_a + dense_b) ** 2
        assert torch.equal(dense_result, expected)
        assert dense_result[0, 1, 3].item() == (7 + 2) ** 2
        assert dense_result[2, 3, 5].item() == 11 ** 2
        assert dense_result[3, 2, 4].item() == 6 ** 2

    def test_mixed_fuse_and_separate(self):
        shape = (4, 5, 6)
        dense1 = torch.zeros(shape, dtype=torch.int64)
        dense1[0, 1, 3] = 1
        dense1[2, 3, 5] = 2
        mct1 = MultiCOOTensor.from_dense(dense1, [[0, 2], [1]])

        dense2_raw = torch.zeros(shape, dtype=torch.int64)
        dense2_raw[0, 1, 3] = 10
        dense2_raw[2, 3, 5] = 20
        mct2 = MultiCOOTensor(
            FactorGraphTensor.from_dense(mct1.conditions.backward(dense2_raw)),
            mct1.conditions,
        )

        dense3 = torch.zeros(shape, dtype=torch.int64)
        dense3[0, 2, 4] = 100
        dense3[3, 2, 4] = 200
        mct3 = MultiCOOTensor.from_dense(dense3, [[0, 1], [2]])

        s = MultiCOOTensorSum([])
        s.add_term(mct1)
        s.add_term(mct2)
        s.add_term(mct3)
        assert len(s.terms) == 2

        result = s.apply_multiplicative(lambda x: x ** 3)
        expected = (
            mct1.conditions.forward(
                mct1.tensor.to_dense() + mct2.tensor.to_dense()
            )
            + mct3.conditions.forward(mct3.tensor.to_dense())
        ) ** 3
        assert torch.equal(_dense_of(result), expected)

    def test_sum_preserves_term_order(self):
        shape = (4, 5, 6)
        dense_first = torch.zeros(shape, dtype=torch.int64)
        dense_first[0, 1, 3] = 1
        dense_second = torch.zeros(shape, dtype=torch.int64)
        dense_second[3, 2, 4] = 2
        mct_first = MultiCOOTensor.from_dense(dense_first, [[0, 2], [1]])
        mct_second = MultiCOOTensor.from_dense(dense_second, [[0, 1], [2]])

        s = MultiCOOTensorSum([])
        s.add_term(mct_first)
        s.add_term(mct_second)
        assert s.terms[0] is mct_first
        assert s.terms[1] is mct_second

    def test_add_term_is_insertion_order_independent_for_disjoint_terms(self):
        shape = (4, 5, 6)
        dense_a = torch.zeros(shape, dtype=torch.int64)
        dense_a[0, 1, 3] = 1
        dense_b = torch.zeros(shape, dtype=torch.int64)
        dense_b[3, 2, 4] = 2
        mct_a = MultiCOOTensor.from_dense(dense_a, [[0, 2], [1]])
        mct_b = MultiCOOTensor.from_dense(dense_b, [[0, 1], [2]])

        s = MultiCOOTensorSum([mct_a, mct_b])
        assert len(s.terms) == 2
        s2 = MultiCOOTensorSum([])
        s2.add_term(mct_a)
        s2.add_term(mct_b)
        assert len(s2.terms) == 2


# --- MultiCOOTensor.sum ---------------------------------------------------


class TestMultiCOOTensorSumReduction:
    def test_sum_all_dims(self):
        conds = _permuted_conds()
        t = torch.tensor([[2, 3], [5, 7]], dtype=torch.int64)
        mct = _mct(t, conds)
        total = mct.sum()
        assert isinstance(total, MultiCOOTensor)
        assert _mct_dense(total).item() == conds.forward(t).sum().item()

    def test_sum_specific_dim(self):
        conds = _triple_permuted_conds()
        t = torch.tensor([3, 4], dtype=torch.int64)
        mct = _mct(t, conds)
        reduced = mct.sum(dim=0)
        assert isinstance(reduced, MultiCOOTensor)
        assert torch.equal(_mct_dense(reduced), conds.forward(t).sum(dim=0))

    def test_sum_keepdim_raises(self):
        conds = _permuted_conds()
        t = torch.tensor([[2, 3], [5, 7]], dtype=torch.int64)
        mct = _mct(t, conds)
        with pytest.raises(NotImplementedError):
            mct.sum(dim=1, keepdim=True)

    def test_sum_permuted_full_condition_elimination(self):
        conds = _permuted_conds()
        t = torch.tensor([[2, 3], [5, 7]], dtype=torch.int64)
        mct = _mct(t, conds)
        reduced = mct.sum(dim=1)
        assert isinstance(reduced, MultiCOOTensor)
        assert torch.equal(_mct_dense(reduced), conds.forward(t).sum(dim=1))

    def test_sum_partial_k2_condition(self):
        conds = _permuted_conds()
        t = torch.tensor([[2, 3], [5, 7]], dtype=torch.int64)
        mct = _mct(t, conds)
        reduced = mct.sum(dim=2)
        assert isinstance(reduced, MultiCOOTensor)
        assert torch.equal(_mct_dense(reduced), conds.forward(t).sum(dim=2))

    def test_sum_partial_k3_coalesce(self):
        conds = _triple_permuted_conds()
        t = torch.tensor([3, 4], dtype=torch.int64)
        mct = _mct(t, conds)
        reduced = mct.sum(dim=2)
        assert torch.equal(_mct_dense(reduced), conds.forward(t).sum(dim=2))

    def test_sum_multiple_dims(self):
        conds = _triple_permuted_conds()
        t = torch.tensor([3, 4], dtype=torch.int64)
        mct = _mct(t, conds)
        reduced = mct.sum(dim=[0, 2])
        assert torch.equal(_mct_dense(reduced), conds.forward(t).sum(dim=(0, 2)))


# --- tensordot ------------------------------------------------------------


class TestMultiCOOTensorTensordot:
    def test_outer_product(self):
        a_conds = COOConditions([_cond([0], [[1, 3]], [4])])
        a_t = torch.tensor([2, 5], dtype=torch.int64)
        a = _mct(a_t, a_conds)

        b_conds = COOConditions([_cond([0], [[0, 2]], [3])])
        b_t = torch.tensor([7, 11], dtype=torch.int64)
        b = _mct(b_t, b_conds)

        result = a.tensordot(b, dims=0)
        expected = torch.tensordot(
            a_conds.forward(a_t), b_conds.forward(b_t), dims=0
        )
        assert torch.equal(_mct_dense(result), expected)

    def test_single_dim_contraction_matching_indices(self):
        shared_values = torch.tensor([1, 4, 6], dtype=torch.int64)
        a_conds = COOConditions([COOCondition(TorchTable([0], [shared_values]), [8])])
        a_t = torch.tensor([2, 3, 5], dtype=torch.int64)
        a = _mct(a_t, a_conds)

        b_conds = COOConditions([COOCondition(TorchTable([0], [shared_values]), [8])])
        b_t = torch.tensor([7, 11, 13], dtype=torch.int64)
        b = _mct(b_t, b_conds)

        result = a.tensordot(b, dims=([0], [0]))
        expected = torch.tensordot(
            a_conds.forward(a_t), b_conds.forward(b_t), dims=([0], [0])
        )
        assert torch.equal(_mct_dense(result), expected)

    def test_contraction_with_batch_dims(self):
        a_conds = COOConditions([
            _cond([0], [[0, 2]], [3]),
            _cond([1], [[1, 5]], [6]),
        ])
        a_t = torch.tensor([[2, 3], [5, 7]], dtype=torch.int64)
        a = _mct(a_t, a_conds)

        b_conds = COOConditions([
            _cond([0], [[1, 5]], [6]),
            _cond([1], [[1, 3]], [4]),
        ])
        b_t = torch.tensor([[11, 13], [17, 19]], dtype=torch.int64)
        b = _mct(b_t, b_conds)

        result = a.tensordot(b, dims=([1], [0]))
        expected = torch.tensordot(
            a_conds.forward(a_t), b_conds.forward(b_t), dims=([1], [0])
        )
        assert torch.equal(_mct_dense(result), expected)

    def test_contraction_partial_index_overlap(self):
        a_conds = COOConditions([_cond([0], [[0, 2]], [5])])
        a = _mct(torch.tensor([1, 2], dtype=torch.int64), a_conds)
        b_conds = COOConditions([_cond([0], [[0, 3]], [5])])
        b = _mct(torch.tensor([1, 2], dtype=torch.int64), b_conds)
        result = a.tensordot(b, dims=([0], [0]))
        expected = torch.tensordot(
            a_conds.forward(torch.tensor([1, 2], dtype=torch.int64)),
            b_conds.forward(torch.tensor([1, 2], dtype=torch.int64)),
            dims=([0], [0]),
        )
        assert torch.equal(_mct_dense(result), expected)

    def test_tensordot_faster_than_dense(self):
        """Sparse-support tensordot avoids the O(D^3) dense contraction."""
        import time

        D = 400
        N = 20
        torch.manual_seed(0)
        shared = torch.randperm(D)[:N].to(torch.int64)
        batch_a = torch.randperm(D)[:N].to(torch.int64)
        batch_b = torch.randperm(D)[:N].to(torch.int64)

        a_conds = COOConditions([
            COOCondition(TorchTable([0], [batch_a]), [D]),
            COOCondition(TorchTable([1], [shared]), [D]),
        ])
        a = _mct(torch.randn(N, N, dtype=torch.float64), a_conds)

        b_conds = COOConditions([
            COOCondition(TorchTable([0], [shared]), [D]),
            COOCondition(TorchTable([1], [batch_b]), [D]),
        ])
        b = _mct(torch.randn(N, N, dtype=torch.float64), b_conds)

        a_dense = _mct_dense(a)
        b_dense = _mct_dense(b)

        _ = a.tensordot(b, dims=([1], [0]))
        _ = torch.tensordot(a_dense, b_dense, dims=([1], [0]))

        start = time.perf_counter()
        result = a.tensordot(b, dims=([1], [0]))
        mct_time = time.perf_counter() - start

        start = time.perf_counter()
        expected = torch.tensordot(a_dense, b_dense, dims=([1], [0]))
        dense_time = time.perf_counter() - start

        assert torch.allclose(_mct_dense(result), expected)
        assert mct_time < dense_time, (
            f"MCT tensordot ({mct_time * 1000:.2f}ms) not faster than "
            f"dense ({dense_time * 1000:.2f}ms)"
        )


# --- gather_indices / scatter_indices -------------------------------------


class TestMultiCOOTensorGatherIndices:
    def test_gather_scatters_single_output_dim(self):
        conds = _permuted_conds()
        t = torch.tensor([[2, 3], [5, 7]], dtype=torch.int64)
        mct = _mct(t, conds)
        gather_idx = torch.tensor(
            [[0, 1, 2, 3, 4], [0, 2, 4, 6, 1]], dtype=torch.int64
        )
        result = mct.gather_indices(gather_idx, dim=1, output_shape=[6, 7])
        dense_in = conds.forward(t)
        expected = _scatter(dense_in, 1, gather_idx, [6, 7])
        assert torch.equal(_mct_dense(result), expected)

    def test_gather_dim_in_multi_dim_condition(self):
        conds = _permuted_conds()
        t = torch.tensor([[2, 3], [5, 7]], dtype=torch.int64)
        mct = _mct(t, conds)
        gather_idx = torch.tensor(
            [[0, 1, 2, 3], [4, 3, 2, 1]], dtype=torch.int64
        )
        result = mct.gather_indices(gather_idx, dim=0, output_shape=[5, 6])
        dense_in = conds.forward(t)
        expected = _scatter(dense_in, 0, gather_idx, [5, 6])
        assert torch.equal(_mct_dense(result), expected)


class TestMultiCOOTensorScatterIndices:
    def test_scatter_single_dim(self):
        conds = _permuted_conds()
        t = torch.tensor([[2, 3], [5, 7]], dtype=torch.int64)
        mct = _mct(t, conds)
        scatter_idx = torch.tensor([[1, 3, 0]], dtype=torch.int64)
        result = mct.scatter_indices(scatter_idx, dims_map=[1])
        dense_in = conds.forward(t)
        expected = _gather(dense_in, [1], scatter_idx, [5])
        assert torch.equal(_mct_dense(result), expected)

    def test_scatter_full_condition(self):
        conds = _permuted_conds()
        t = torch.tensor([[2, 3], [5, 7]], dtype=torch.int64)
        mct = _mct(t, conds)
        scatter_idx = torch.tensor([[0, 2], [3, 5]], dtype=torch.int64)
        result = mct.scatter_indices(scatter_idx, dims_map=[0, 2])
        dense_in = conds.forward(t)
        expected = _gather(dense_in, [0, 2], scatter_idx, [4, 6])
        assert torch.equal(_mct_dense(result), expected)

    def test_scatter_non_support_targets_are_zero(self):
        conds = COOConditions([_cond([0], [[2, 7]], [10])])
        t = torch.tensor([9, 13], dtype=torch.int64)
        mct = _mct(t, conds)
        scatter_idx = torch.tensor([[7, 5, 2]], dtype=torch.int64)
        result = mct.scatter_indices(scatter_idx, dims_map=[0])
        dense_in = conds.forward(t)
        expected = _gather(dense_in, [0], scatter_idx, [10])
        assert torch.equal(_mct_dense(result), expected)

    def test_gather_scatter_roundtrip(self):
        conds = COOConditions([_eye_cond([0], length=3)])
        t = torch.tensor([9, 13, 17], dtype=torch.int64)
        mct = _mct(t, conds)
        gather_idx = torch.tensor(
            [[0, 1, 2], [1, 3, 0]], dtype=torch.int64
        )
        gathered = mct.gather_indices(gather_idx, dim=0, output_shape=[4, 5])
        back = gathered.scatter_indices(gather_idx, dims_map=[0, 1])
        assert torch.equal(_mct_dense(back), conds.forward(t))
