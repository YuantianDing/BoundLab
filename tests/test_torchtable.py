import pytest
import torch

from boundlab.sparse.table import TorchTable


def idx(values):
    return torch.tensor(values, dtype=torch.int64)


# --- construction ---------------------------------------------------------

class TestConstruction:
    def test_concrete_columns_infers_length(self):
        t = TorchTable([0, 3], [idx([1, 2, 3]), idx([4, 5, 6])])
        assert t.length == 3
        assert t.columns == [0, 3]
        assert not t.is_sorted
        assert not t.is_unique

    def test_none_column_requires_length(self):
        with pytest.raises(AssertionError):
            TorchTable([0], [None])
        t = TorchTable([0], [None], length=4)
        assert t.length == 4
        assert t.is_all_indices()

    def test_mixed_columns(self):
        t = TorchTable([5, 2], [None, idx([7, 8, 9])])
        assert t.length == 3
        assert not t.is_all_indices()

    def test_length_mismatch_raises(self):
        with pytest.raises(AssertionError):
            TorchTable([0, 1], [idx([1, 2]), idx([1, 2, 3])])

    def test_duplicate_columns_raise(self):
        with pytest.raises(AssertionError):
            TorchTable([3, 3], [idx([1, 2]), idx([3, 4])])


# --- materialize ----------------------------------------------------------

class TestMaterialize:
    def test_all_indices(self):
        t = TorchTable([2, 5], [None, None], length=3)
        mat = t.materialize()
        expected = torch.stack([torch.arange(3), torch.arange(3)], dim=1)
        assert torch.equal(mat, expected)

    def test_mixed(self):
        t = TorchTable([0, 1], [idx([9, 8, 7]), None])
        mat = t.materialize()
        assert torch.equal(mat[:, 0], idx([9, 8, 7]))
        assert torch.equal(mat[:, 1], torch.arange(3))

    def test_zero_columns(self):
        t = TorchTable([], [], length=5)
        mat = t.materialize()
        assert mat.shape == (5, 0)


# --- sort_dedup -----------------------------------------------------------

class TestSortDedup:
    def test_unsorted_unique(self):
        t = TorchTable([0, 1], [idx([3, 1, 2]), idx([30, 10, 20])])
        new, inv = t.sort_dedup()
        assert new.is_sorted and new.is_unique
        assert new.length == 3
        assert torch.equal(new.materialize(),
                           torch.tensor([[1, 10], [2, 20], [3, 30]]))
        # inverse[i] = new position of old row i. Old rows (3,30)→2, (1,10)→0, (2,20)→1.
        assert torch.equal(inv, idx([2, 0, 1]))

    def test_sort_dedup_collapses_duplicates(self):
        t = TorchTable([0], [idx([5, 2, 5, 2, 8])])
        new, inv = t.sort_dedup()
        assert new.length == 3
        assert torch.equal(new.materialize().squeeze(1), idx([2, 5, 8]))
        # old rows: (5,2,5,2,8) → new rows (1,0,1,0,2)
        assert torch.equal(inv, idx([1, 0, 1, 0, 2]))

    def test_already_sorted_unique_returns_none_inverse(self):
        t = TorchTable([0], [idx([1, 2, 3])])
        t.is_sorted = True
        t.is_unique = True
        new, inv = t.sort_dedup()
        assert inv is None
        assert new is t

    def test_compresses_to_index_column(self):
        t = TorchTable([0, 1], [idx([2, 0, 1]), idx([20, 0, 10])])
        new, _ = t.sort_dedup()
        # After sorting by col 0: rows become (0,0), (1,10), (2,20)
        # Col 0 equals arange(3) → should be compressed to None.
        assert new.data[0] is None
        assert new.data[1] is not None
        assert torch.equal(new.data[1], idx([0, 10, 20]))

    def test_empty_table(self):
        t = TorchTable([0], [idx([], )])
        new, inv = t.sort_dedup()
        assert new.length == 0
        assert inv.shape == (0,)

    def test_zero_columns(self):
        t = TorchTable([], [], length=4)
        new, inv = t.sort_dedup()
        # With no columns, all rows collapse to 1.
        assert new.length == 1
        assert torch.equal(inv, torch.zeros(4, dtype=torch.int64))


# --- dedup ----------------------------------------------------------------

class TestDedup:
    def test_dedup_requires_sorted(self):
        t = TorchTable([0], [idx([3, 1, 2])])
        with pytest.raises(AssertionError):
            t.dedup()

    def test_dedup_sorted_already_unique(self):
        t = TorchTable([0], [idx([1, 2, 3])])
        t.is_sorted = True
        new, inv = t.dedup()
        assert inv is None
        assert new.length == 3

    def test_dedup_collapses(self):
        t = TorchTable([0, 1], [idx([1, 1, 2, 2, 3]), idx([10, 10, 20, 20, 30])])
        t.is_sorted = True
        new, inv = t.dedup()
        assert new.length == 3
        assert new.is_sorted and new.is_unique
        assert torch.equal(new.materialize(),
                           torch.tensor([[1, 10], [2, 20], [3, 30]]))
        assert torch.equal(inv, idx([0, 0, 1, 1, 2]))


# --- merge ----------------------------------------------------------------

class TestMerge:
    def _sorted_unique(self, columns, data, length=None):
        t = TorchTable(columns, data, length=length)
        t, _ = t.sort_dedup()
        return t

    def test_merge_single_returns_self(self):
        t = self._sorted_unique([0], [idx([1, 2])])
        assert TorchTable.merge([t]) is t

    def test_fast_path_all_indices_shared(self):
        a = TorchTable([0, 1], [None, None], length=5)
        a.is_sorted = True
        a.is_unique = True
        b = TorchTable([0, 2], [None, None], length=3)
        b.is_sorted = True
        b.is_unique = True
        # Shared col [0] is index in both — fast path.
        out = TorchTable.merge([a, b])
        assert out.length == 3
        assert set(out.columns) == {0, 1, 2}
        # All three cols should be index columns: shared col 0 is arange(3),
        # col 1 from a sliced arange(5)[:3] = arange(3), col 2 from b is arange(3).
        assert out.is_all_indices()

    def test_fast_path_preserves_non_shared_concrete_columns(self):
        a = TorchTable([0, 1], [None, idx([9, 8, 7, 6])], length=4)
        a.is_sorted = True
        a.is_unique = True
        b = TorchTable([0, 2], [None, idx([30, 20])], length=2)
        b.is_sorted = True
        b.is_unique = True
        out = TorchTable.merge([a, b])
        assert out.length == 2
        # col 0 shared index → arange(2)
        # col 1 from a sliced → [9, 8]
        # col 2 from b → [30, 20]
        col_of = {c: out.data[out.columns.index(c)] for c in out.columns}
        assert col_of[0] is None
        assert torch.equal(col_of[1], idx([9, 8]))
        assert torch.equal(col_of[2], idx([30, 20]))

    def test_slow_path_concrete_shared(self):
        # Shared col 0 with concrete values — slow path via pandas join.
        a = TorchTable([0, 1], [idx([1, 2, 3]), idx([10, 20, 30])])
        a, _ = a.sort_dedup()
        b = TorchTable([0, 2], [idx([2, 3, 4]), idx([200, 300, 400])])
        b, _ = b.sort_dedup()
        out = TorchTable.merge([a, b])
        assert out.length == 2  # rows where col 0 ∈ {2, 3}
        assert set(out.columns) == {0, 1, 2}
        # Build a dict by col0 value → (col1, col2) for unordered comparison.
        mat = out.materialize()
        col0 = mat[:, out.columns.index(0)]
        col1 = mat[:, out.columns.index(1)]
        col2 = mat[:, out.columns.index(2)]
        rows = sorted(zip(col0.tolist(), col1.tolist(), col2.tolist()))
        assert rows == [(2, 20, 200), (3, 30, 300)]

    def test_requires_sorted_unique(self):
        a = TorchTable([0], [idx([1, 2, 3])])
        b = TorchTable([0], [idx([2, 3])])
        with pytest.raises(AssertionError):
            TorchTable.merge([a, b])

    def test_and_operator(self):
        a = TorchTable([0], [None], length=4)
        a.is_sorted = True
        a.is_unique = True
        b = TorchTable([0, 1], [None, None], length=3)
        b.is_sorted = True
        b.is_unique = True
        out = a & b
        assert out.length == 3
        assert set(out.columns) == {0, 1}


# --- index ----------------------------------------------------------------

class TestIndex:
    def test_basic_index(self):
        t = TorchTable([0, 1], [idx([1, 2, 3, 4]), idx([10, 20, 30, 40])])
        t, _ = t.sort_dedup()
        q = TorchTable([1, 0], [idx([30, 10]), idx([3, 1])])
        out = t.index(q)
        # t after sort: (1,10), (2,20), (3,30), (4,40). q's rows (col order swapped):
        # row0: col1=30, col0=3 → (3, 30) → pos 2
        # row1: col1=10, col0=1 → (1, 10) → pos 0
        assert torch.equal(out, idx([2, 0]))

    def test_requires_same_columns(self):
        a = TorchTable([0], [idx([1, 2])])
        a, _ = a.sort_dedup()
        b = TorchTable([1], [idx([1])])
        with pytest.raises(AssertionError):
            a.index(b)

    def test_requires_self_unique(self):
        a = TorchTable([0], [idx([1, 1, 2])])
        b = TorchTable([0], [idx([1])])
        with pytest.raises(AssertionError):
            a.index(b)


# --- filter_columns -------------------------------------------------------

class TestFilterColumns:
    def test_identity_projection(self):
        t = TorchTable([0, 1], [idx([1, 2, 3]), idx([10, 20, 30])])
        t, _ = t.sort_dedup()
        out, inv = t.filter_columns([0, 1])
        assert inv is None
        assert out.length == 3

    def test_reorder_columns(self):
        t = TorchTable([0, 1], [idx([1, 2, 3]), idx([10, 20, 30])])
        t, _ = t.sort_dedup()
        out, inv = t.filter_columns([1, 0])
        # Column order differs → sort_dedup runs; inverse not None.
        assert out.columns == [1, 0]
        assert inv is not None

    def test_drop_non_coalescing(self):
        # Drop col 1 where col 0 alone is already unique.
        t = TorchTable([0, 1], [idx([1, 2, 3]), idx([10, 20, 30])])
        t, _ = t.sort_dedup()
        out, inv = t.filter_columns([0])
        assert out.length == 3
        assert inv is not None
        # Inverse should be a permutation (identity here, col 0 already sorted).
        assert torch.equal(inv, idx([0, 1, 2]))

    def test_drop_coalesces(self):
        t = TorchTable([0, 1], [idx([1, 1, 2, 2, 3]), idx([10, 20, 30, 40, 50])])
        t, _ = t.sort_dedup()
        out, inv = t.filter_columns([0])
        assert out.length == 3
        assert torch.equal(out.materialize().squeeze(1), idx([1, 2, 3]))
        assert torch.equal(inv, idx([0, 0, 1, 1, 2]))

    def test_subset_assertion(self):
        t = TorchTable([0], [idx([1, 2])])
        with pytest.raises(AssertionError):
            t.filter_columns([99])
