from collections import defaultdict
from typing import Optional

import torch

from boundlab.sparse.ops import list_index_unique, table_join_sorted
from .dim import Dim
Indices = torch.Tensor
"""A 1D tensor of int indices, used for typing."""


class TorchTable:
    """SQL-style table of integer columns, labelled by :class:`Dim` objects.

    A column's data can be ``None``, meaning it's an *index column* conceptually equal to
    ``torch.arange(length)``. Supports sorting, deduplication, inner-join merging on shared
    columns, and row indexing. ONNX-exportable via ``torch.onnx.ops.symbolic`` (TODO).
    """

    def __init__(
        self,
        columns: list[Dim],
        data: list[Optional[Indices]],
        length: Optional[int] = None,
        is_sorted: bool = False,
        is_unique: bool = False,
    ):
        assert len(columns) == len(data), "Must provide data for each column."
        assert len(set(columns)) == len(columns), "Column names must be unique."
        self.columns: list[Dim] = list(columns)
        self.data: list[Optional[Indices]] = data
        set_len = set(int(d.shape[0]) for d in self.data if d is not None)
        if length is not None:
            set_len.add(length)
        assert len(set_len) == 1, "All columns must have the same length as each other and the provided length."
        self.length: int = set_len.pop()

        for i, dat in enumerate(self.data):
            if dat is not None:
                assert dat.dim() == 1, "Column data must be 1D."
                assert int(dat.shape[0]) == self.length, \
                        "All columns must have the same length."
            elif data is None:
                assert self.columns[i].length == self.length, f"Index column length must match table length, but column {self.columns[i]} has length {self.columns[i].length} and table has length {self.length}."
        assert self.length is not None, \
            "Must provide at least one column of data or specify length."
        self.is_sorted = is_sorted
        self.is_unique = is_unique
        if (len(self.data) > 0 and all(dat is None for dat in self.data)) or (
            len(self.data) == 0 and self.length <= 1
        ):
            self.is_sorted = True
            self.is_unique = True
    
    def items(self):
        """Iterator over (column_name, column_data) pairs."""
        return zip(self.columns, self.data)

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

    def column_data(self, name: Dim) -> Optional[Indices]:
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
    def reduce_table_merge(tables: list["TorchTable"]) -> list["TorchTable"]:
        """Reduce a list of tables by merging those that share the same values in the given columns."""
        cols = defaultdict[Dim, list[tuple[int, int]]](list)
        dims = []
        for i, t1 in enumerate(tables):
            for j, c in enumerate(t1.columns):
                cols[c].append((i, j))
                if t1.data[j] is None and c not in dims:
                    dims.append(c)
        for dim in dims:
            a = TorchTable.reduce_table_merge_inner(dim, tables, cols[dim])
            if a is not None:
                return TorchTable.reduce_table_merge(a)
        return tables


            
    @staticmethod
    def reduce_table_merge_inner(dim: Dim, tables: list["TorchTable"], cols: list[tuple[int, int]]) -> list["TorchTable"]:
        remaining = [t for i, t in enumerate(tables) if all(i != idx for idx, _ in cols)]
        related = [tables[idx] for idx, j in cols if tables[idx].data[j] is None]
        related_table = TorchTable.combine(related)
        if related_table is None:
            return None
        main = [tables[idx] for idx, j in cols if tables[idx].data[j] is not None]
        if len(related) == 1 and len(main) == 0:
            return None
        if len(main) == 0:
            return remaining + [related_table]
        
        main_table = TorchTable.merge(main, no_reduce=True)

        indices: Indices = main_table.column_data(dim)
        related_table = related_table[indices]
        combined = TorchTable.combine([related_table, main_table])
        if combined is None:
            return None
        return [combined] + remaining

    @staticmethod
    def combine(tables: list["TorchTable"]) -> "TorchTable":
        """Combine same-row tables, trimming arange-backed index columns to the shared prefix."""
        assert len(tables) > 0, "combine() requires at least one table."
        length = min(t.length for t in tables)
        table = dict()
        for t in tables:
            for c, d in t.items():
                data = d[:length] if d is not None else None
                if c in table:
                    existing = table[c]
                    existing_mat = (
                        torch.arange(length, dtype=torch.int64)
                        if existing is None
                        else existing.to(torch.int64)
                    )
                    data_mat = (
                        torch.arange(length, dtype=torch.int64)
                        if data is None
                        else data.to(torch.int64)
                    )
                    if not torch.equal(existing_mat, data_mat):
                        return None
                    continue
                table[c] = data
        result = TorchTable(list(table.keys()), list(table.values()), length=length)
        result.is_unique = any(t.is_unique for t in tables)
        return result
        

    @staticmethod
    def merge(tables: list["TorchTable"], no_reduce: bool = False) -> "TorchTable":
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
        assert all(t.is_unique for t in tables), \
            "All tables must be sorted and unique before merging."
        
        if not no_reduce:
            tables = TorchTable.reduce_table_merge(tables)
        if len(tables) == 1:
            return tables[0]

        # Slow path — remap columns to dense indices and call table_join_sorted.
        all_col_names = sorted(set(k for t in tables for k, _ in t.items()))
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
            result_tensor[:, i]
            for i in range(len(all_col_names))
        ]
        return TorchTable(list(all_col_names), new_data, length=new_length, is_unique=True)

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
        self, columns: list[Dim]
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
    
    def replace_columns(self, dim_mapping: dict[Dim, Dim]) -> "TorchTable":
        """In-place rename columns according to the provided mapping."""
        new_columns = [dim_mapping.get(c, c) for c in self.columns]
        assert len(set(new_columns)) == len(new_columns), "Renamed columns must be unique."
        result = TorchTable(new_columns, self.data, length=self.length)
        result.is_sorted = self.is_sorted
        result.is_unique = self.is_unique
        return result
        
    
    def permute_(self, permutation: list[int]):
        """In-place permute columns to match the given order."""
        assert set(permutation) == set(range(len(self.columns))), "Permutation must be a rearrangement of all columns."
        self.columns = [self.columns[i] for i in permutation]
        self.data = [self.data[i] for i in permutation]
        self.is_sorted = False
    def __getitem__(self, idx: Indices) -> "TorchTable":
        """Return a new table with the same columns but only the row at index ``idx``."""
        new_data = []
        for dat in self.data:
            if dat is None:
                new_data.append(idx)
            else:
                new_data.append(dat[idx])
        return TorchTable(list(self.columns), new_data, length=idx.shape[0])
__all__ = ["TorchTable", "Indices"]
