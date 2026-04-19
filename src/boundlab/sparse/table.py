from typing import Optional, Union

import torch

from boundlab.sparse.ops import list_index_unique, table_join_sorted

Indices = torch.Tensor
"""A 1D tensor of int indices, used for typing."""


class TorchTable:
    """SQL-style table of integer columns, labelled by integer names (MCT dim IDs).

    A column's data can be ``None``, meaning it's an *index column* conceptually equal to
    ``torch.arange(length)``. Supports sorting, deduplication, inner-join merging on shared
    columns, and row indexing. ONNX-exportable via ``torch.onnx.ops.symbolic`` (TODO).
    """

    def __init__(
        self,
        columns: list[int],
        data: list[Optional[Indices]],
        length: Optional[int] = None,
    ):
        assert len(columns) == len(data), "Must provide data for each column."
        assert len(set(columns)) == len(columns), "Column names must be unique."
        self.columns: list[int] = list(columns)
        self.data: list[Optional[Indices]] = list(data)
        self.length: Optional[int] = length
        for dat in self.data:
            if dat is not None:
                assert dat.dim() == 1, "Column data must be 1D."
                if self.length is None:
                    self.length = int(dat.shape[0])
                else:
                    assert int(dat.shape[0]) == self.length, \
                        "All columns must have the same length."
        assert self.length is not None, \
            "Must provide at least one column of data or specify length."
        self.is_sorted = False
        self.is_unique = False

    def __len__(self) -> int:
        return self.length

    def __repr__(self) -> str:
        flags = []
        if self.is_sorted:
            flags.append("sorted")
        if self.is_unique:
            flags.append("unique")
        flags_str = f"[{','.join(flags)}]" if flags else ""
        return (
            f"TorchTable(columns={self.columns}, length={self.length}{flags_str})"
        )

    def is_all_indices(self) -> bool:
        """True when every column is an index column (``data[k] is None``)."""
        return all(dat is None for dat in self.data)

    def column_data(self, name: int) -> Optional[Indices]:
        """Return the raw ``data`` entry for the given column name."""
        return self.data[self.columns.index(name)]

    def materialize(self) -> torch.Tensor:
        """Return a dense ``(length, num_cols)`` int64 tensor with all columns concrete.

        Index columns are expanded via ``torch.arange``.
        """
        if len(self.columns) == 0:
            return torch.zeros((self.length, 0), dtype=torch.int64)
        cols = []
        for dat in self.data:
            if dat is None:
                cols.append(torch.arange(self.length, dtype=torch.int64))
            else:
                cols.append(dat.to(torch.int64))
        return torch.stack(cols, dim=1)

    @staticmethod
    def _maybe_compress(col: torch.Tensor, length: int) -> Optional[torch.Tensor]:
        """Return ``None`` if ``col`` equals ``arange(length)``, else the tensor itself.

        Used after sort/dedup/merge to preserve the index-column compression
        whenever the materialised column happens to be ``0..length-1``.
        """
        if col.shape[0] != length:
            return col
        arange = torch.arange(length, dtype=col.dtype)
        if torch.equal(col, arange):
            return None
        return col

    def sort_dedup(self) -> tuple["TorchTable", Optional[Indices]]:
        """Sort rows lexicographically by all columns and drop duplicates.

        Returns ``(new_table, inverse)`` where ``inverse[i]`` is the new row index
        of original row ``i``. Returns ``inverse=None`` when the table is already
        sorted+unique (the inverse would be ``arange(length)``).
        """
        if self.is_sorted and self.is_unique:
            return self, None

        # Edge case: no columns — every row is the same row, collapse to 0 or 1.
        if len(self.columns) == 0:
            new_length = 1 if self.length > 0 else 0
            new = TorchTable(list(self.columns), [], length=new_length)
            new.is_sorted = True
            new.is_unique = True
            inverse = torch.zeros(self.length, dtype=torch.int64)
            return new, inverse

        mat = self.materialize()  # (length, K) int64
        # Lexicographic sort via successive stable sorts from right to left.
        order = torch.arange(self.length, dtype=torch.int64)
        for k in reversed(range(len(self.columns))):
            col = mat[:, k][order]
            _, sub_order = torch.sort(col, stable=True)
            order = order[sub_order]
        sorted_mat = mat[order]

        if self.length <= 1:
            mask = torch.ones(self.length, dtype=torch.bool)
        else:
            diff = (sorted_mat[1:] != sorted_mat[:-1]).any(dim=1)
            mask = torch.cat([torch.ones(1, dtype=torch.bool), diff])

        unique_mat = sorted_mat[mask]
        new_length = int(mask.sum().item())

        # Position of each sorted row in the unique output; -1 cumsum trick
        # gives [0, 0, 1, 1, 2, ...] for a mask like [T, F, T, F, T].
        pos_in_unique_sorted = mask.to(torch.int64).cumsum(dim=0) - 1
        inverse = torch.empty(self.length, dtype=torch.int64)
        inverse[order] = pos_in_unique_sorted

        new_data = [
            TorchTable._maybe_compress(unique_mat[:, k], new_length)
            for k in range(len(self.columns))
        ]
        new = TorchTable(list(self.columns), new_data, length=new_length)
        new.is_sorted = True
        new.is_unique = True
        return new, inverse

    def dedup(self) -> tuple["TorchTable", Optional[Indices]]:
        """Deduplicate rows assuming the table is already sorted.

        Returns ``(new_table, inverse)`` as in :meth:`sort_dedup`.
        """
        assert self.is_sorted, "dedup() requires is_sorted=True."
        if self.is_unique:
            return self, None
        if len(self.columns) == 0 or self.length <= 1:
            return self.sort_dedup()

        mat = self.materialize()
        diff = (mat[1:] != mat[:-1]).any(dim=1)
        if bool(diff.all().item()):
            self.is_unique = True
            return self, None
        mask = torch.cat([torch.ones(1, dtype=torch.bool), diff])
        unique_mat = mat[mask]
        new_length = int(mask.sum().item())
        inverse = mask.to(torch.int64).cumsum(dim=0) - 1
        new_data = [
            TorchTable._maybe_compress(unique_mat[:, k], new_length)
            for k in range(len(self.columns))
        ]
        new = TorchTable(list(self.columns), new_data, length=new_length)
        new.is_sorted = True
        new.is_unique = True
        return new, inverse

    @staticmethod
    def merge(tables: list["TorchTable"]) -> "TorchTable":
        """Inner-join a list of sorted+unique tables on their shared columns.

        Fast path: when every shared column is an index column in every input,
        the join degenerates to slicing each non-shared column to
        ``min(length_i)`` — no materialisation needed.

        Slow path: materialise each table and call :func:`ops.table_join_sorted`
        after remapping column names onto a dense ``0..N-1`` range (the op
        requires dense column indices).
        """
        assert len(tables) > 0, "merge() requires at least one table."
        assert all(isinstance(t, TorchTable) for t in tables)
        assert all(t.is_sorted and t.is_unique for t in tables), \
            "All tables must be sorted and unique before merging."
        if len(tables) == 1:
            return tables[0]

        col_sets = [set(t.columns) for t in tables]
        shared: set[int] = set(col_sets[0])
        for s in col_sets[1:]:
            shared &= s

        all_shared_are_indices = len(shared) > 0 and all(
            t.data[t.columns.index(c)] is None for t in tables for c in shared
        )
        if all_shared_are_indices:
            min_length = min(t.length for t in tables)
            new_columns: list[int] = []
            new_data: list[Optional[Indices]] = []
            seen: set[int] = set()
            for t in tables:
                for name, dat in zip(t.columns, t.data):
                    if name in seen:
                        continue
                    seen.add(name)
                    new_columns.append(name)
                    if name in shared:
                        new_data.append(None)
                    elif dat is None:
                        new_data.append(None)
                    else:
                        sliced = dat[:min_length]
                        new_data.append(
                            TorchTable._maybe_compress(sliced, min_length)
                        )
            result = TorchTable(new_columns, new_data, length=min_length)
            # Sorted+unique only guaranteed when each table individually was;
            # the index-column fast-path preserves sort+unique on shared cols.
            result.is_sorted = True
            result.is_unique = True
            return result

        # Slow path — remap columns to dense indices and call table_join_sorted.
        all_col_names = sorted(set().union(*col_sets))
        name_to_idx = {name: i for i, name in enumerate(all_col_names)}
        args: list = []
        for t in tables:
            mat = t.materialize()
            remapped_cols = [name_to_idx[c] for c in t.columns]
            args.append(mat)
            args.append(remapped_cols)
        result_tensor = table_join_sorted(*args)
        new_length = int(result_tensor.shape[0])
        new_data = [
            TorchTable._maybe_compress(result_tensor[:, i], new_length)
            for i in range(len(all_col_names))
        ]
        return TorchTable(list(all_col_names), new_data, length=new_length)

    def __and__(self, other: "TorchTable") -> "TorchTable":
        return TorchTable.merge([self, other])

    def index(self, other: "TorchTable") -> Indices:
        """Return ``idx`` such that ``self[idx]`` equals ``other`` row-wise.

        Requires the two tables to share the same column set and ``self`` to be
        unique; every row of ``other`` must appear exactly once in ``self``.
        """
        assert self.is_unique, "self must be unique to be indexed."
        assert set(self.columns) == set(other.columns), \
            f"Column mismatch: {self.columns} vs {other.columns}"
        aligned_other_data = [
            other.data[other.columns.index(c)] for c in self.columns
        ]
        aligned_other = TorchTable(
            list(self.columns), aligned_other_data, length=other.length
        )
        return list_index_unique(self.materialize(), aligned_other.materialize())

    def filter_columns(
        self, columns: list[int]
    ) -> tuple["TorchTable", Optional[Indices]]:
        """Project onto ``columns`` (possibly reordering/dropping) then sort+dedup.

        Returns ``(new_table, inverse)``:
        - ``new_table`` has exactly ``columns`` as its columns, sorted+unique.
        - ``inverse[i]`` is the new-row index of the old row ``i``.
        - ``inverse`` is ``None`` when no coalescence or reordering occurred —
          i.e. ``new_table`` is identical to ``self`` (same columns in same order,
          already sorted+unique).
        """
        assert set(columns).issubset(set(self.columns)), \
            f"Columns {columns} not a subset of {self.columns}"
        assert len(set(columns)) == len(columns), "columns must be unique."

        positions = [self.columns.index(c) for c in columns]
        projected_data = [self.data[p] for p in positions]
        projected = TorchTable(list(columns), projected_data, length=self.length)

        if (
            columns == self.columns
            and self.is_sorted
            and self.is_unique
        ):
            projected.is_sorted = True
            projected.is_unique = True
            return projected, None

        return projected.sort_dedup()


__all__ = ["TorchTable", "Indices"]
