from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union

import torch

from boundlab import utils
from boundlab.sparse.table import TorchTable
from .factors import (
    FactorGraphTensor,
    FactorTensor,
    _gather_axes_via_indices,
    _scatter_axis_via_indices,
)
from .ops import list_index_unique, list_is_subset

Tensorish = Union[torch.Tensor, FactorGraphTensor]


def _relabel_table(table: TorchTable, new_columns: list[int]) -> TorchTable:
    """Return a view of ``table`` with column names replaced by ``new_columns``."""
    assert len(new_columns) == len(table.columns)
    out = TorchTable(list(new_columns), list(table.data), length=table.length)
    out.is_sorted = table.is_sorted
    out.is_unique = table.is_unique
    return out


def _all_values_cond(dim_label: int, size: int) -> "COOCondition":
    """K=1 condition covering every value 0..size-1 of ``dim_label`` (index column)."""
    table = TorchTable([dim_label], [None], length=size)
    table.is_sorted = True
    table.is_unique = True
    return COOCondition(table, [size])


@dataclass
class COOCondition:
    """Structural sparsity constraint on a group of MCT output dims.

    ``table.columns`` are the MCT output-dim IDs covered by this condition;
    each row of ``table`` is one allowed value tuple. ``output_shape[k]`` is
    the per-axis size (exclusive upper bound) of the k-th column.

    ``symbolic_subsets`` tracks ids of conditions known to be subsets of this
    condition (for export-safe subset checks when data is not available).
    """

    table: TorchTable
    output_shape: list[int]
    symbolic_subsets: list[int] = field(default_factory=list)

    def __post_init__(self):
        assert len(self.output_shape) == len(self.table.columns), \
            f"output_shape {self.output_shape} must align with table columns {self.table.columns}"
        self.output_shape = [int(s) for s in self.output_shape]

    @property
    def dims(self) -> list[int]:
        return self.table.columns

    @property
    def K(self) -> int:
        return len(self.table.columns)

    @property
    def N(self) -> int:
        return self.table.length

    def is_empty(self) -> bool:
        return self.N == 0

    def canonical(self) -> "COOCondition":
        """Return a copy with ``table`` sorted+deduped (idempotent when already canonical).

        Result is cached on ``self`` so repeated callers in hot paths (``all``,
        ``is_subset``, ``&``) don't re-sort the same table.
        """
        if self.table.is_sorted and self.table.is_unique:
            return self
        cached = getattr(self, "_canonical_cache", None)
        if cached is not None:
            return cached
        new_table, _ = self.table.sort_dedup()
        result = COOCondition(
            new_table, list(self.output_shape), list(self.symbolic_subsets)
        )
        object.__setattr__(self, "_canonical_cache", result)
        return result

    def forward(self, compressed: torch.Tensor, axis: int) -> torch.Tensor:
        """Scatter ``compressed``'s ``axis`` (length N) into K axes of sizes ``output_shape``.

        The K new axes replace ``axis`` in place; their order matches
        ``table.columns``.
        """
        if axis < 0:
            axis += compressed.ndim
        assert 0 <= axis < compressed.ndim
        assert int(compressed.shape[axis]) == self.N
        indices = self.table.materialize().T.contiguous()  # (K, N)
        return _scatter_axis_via_indices(compressed, axis, indices, self.output_shape)

    def backward(self, dense: torch.Tensor, dims_map: list[int]) -> torch.Tensor:
        """Gather K axes (at ``dims_map``) into one compressed axis of length N."""
        assert len(dims_map) == self.K
        indices = self.table.materialize().T.contiguous()  # (K, N)
        return _gather_axes_via_indices(dense, dims_map, indices, self.output_shape)

    @staticmethod
    def all(*conds: "COOCondition") -> "COOCondition":
        """Merge conditions sharing at least one dim via inner join on shared dims."""
        assert len(conds) >= 1
        col_size: dict[int, int] = {}
        for c in conds:
            for col, size in zip(c.table.columns, c.output_shape):
                if col in col_size:
                    assert col_size[col] == int(size), \
                        f"Dim {col} has inconsistent sizes {col_size[col]} vs {size}"
                else:
                    col_size[col] = int(size)

        # Drop all-indices conds whose col set is a subset of some other cond's
        # cols — their rows are ``arange(length)`` and the length is already
        # validated via ``col_size``, so they contribute no filtering. Operate
        # on the originals (no sort) since the all-indices check is
        # order-invariant.
        kept: list[COOCondition] = []
        for i, c in enumerate(conds):
            if c.table.is_all_indices() and any(
                j != i and set(c.table.columns).issubset(set(other.table.columns))
                for j, other in enumerate(conds)
            ):
                continue
            kept.append(c)
        if not kept:
            kept = [conds[0]]

        # Identical-data fast path: if all remaining tables share the same cols
        # and row data, return the first as-is (no sort, no join). Keeping the
        # shared tensor reference lets ``shrink_to`` short-circuit via ``is``.
        def _same_data(a: TorchTable, b: TorchTable) -> bool:
            if a.columns != b.columns or a.length != b.length:
                return False
            for da, db in zip(a.data, b.data):
                if da is db:
                    continue
                if da is None or db is None:
                    return False
                if not torch.equal(da, db):
                    return False
            return True

        if len(kept) == 1 or all(
            _same_data(kept[0].table, t.table) for t in kept[1:]
        ):
            chosen = kept[0]
            output_shape = [col_size[col] for col in chosen.table.columns]
            result = COOCondition(chosen.table, output_shape)
            cached = getattr(chosen, "_canonical_cache", None)
            if cached is not None:
                object.__setattr__(result, "_canonical_cache", cached)
            for c in conds:
                c.symbolic_subsets.append(id(result))
            return result

        canon = [c.canonical() for c in kept]
        merged = TorchTable.merge([c.table for c in canon])
        if not (merged.is_sorted and merged.is_unique):
            merged, _ = merged.sort_dedup()
        output_shape = [col_size[col] for col in merged.columns]
        result = COOCondition(merged, output_shape)
        for c in conds:
            c.symbolic_subsets.append(id(result))
        return result

    def __and__(self, other: "COOCondition") -> "COOCondition":
        return COOCondition.all(self, other)

    def is_subset(self, other: "COOCondition") -> bool:
        """``self ⊆ other``: self restricted to other's dims is a subset of other's rows.

        Requires ``other.dims ⊆ self.dims``. Returns False otherwise.
        """
        if not set(other.table.columns).issubset(set(self.table.columns)):
            return False
        if id(self) in other.symbolic_subsets:
            return True
        self_canon = self.canonical()
        other_canon = other.canonical()
        projected, _ = self_canon.table.filter_columns(list(other_canon.table.columns))
        return bool(list_is_subset(projected.materialize(), other_canon.table.materialize()))

    def is_subset_symbolic(self, other: "COOCondition") -> bool:
        """Export-safe subset check — avoids data-dependent ops when possible."""
        if (
            self.table.is_all_indices()
            and other.table.is_all_indices()
            and self.table.length <= other.table.length
        ):
            return set(other.table.columns).issubset(set(self.table.columns))
        return id(self) in other.symbolic_subsets

    def partial_dim_sum(
        self, dims_to_sum: list[int]
    ) -> tuple["COOCondition", Optional[torch.Tensor]]:
        """Drop ``dims_to_sum`` from this condition; return ``(new_cond, inverse)``.

        ``inverse[i]`` is the new-row index of old row ``i`` (for scatter-add
        coalescence on the compressed FGT axis); ``None`` when no coalescence
        occurred.
        """
        kept = [d for d in self.table.columns if d not in set(dims_to_sum)]
        new_table, inverse = self.canonical().table.filter_columns(kept)
        new_output_shape = [
            self.output_shape[self.table.columns.index(d)]
            for d in new_table.columns
        ]
        return COOCondition(new_table, new_output_shape), inverse


class COOConditions(list[COOCondition]):
    """Ordered list of dim-disjoint conditions covering ``range(dim())``."""

    def __init__(self, conditions: list[COOCondition] = ()):
        super().__init__(conditions)
        self.sort(key=lambda c: c.dims)
        check: set[int] = set()
        for c in self:
            for d in c.dims:
                assert d >= 0, f"Condition dimensions must be non-negative: got {d}"
                if d in check:
                    raise ValueError(f"Duplicate dimension across conditions: {d}")
                check.add(d)
        assert len(check) == self.dim(), \
            "Conditions must cover range(dim()) without gaps or overlap"

    def dim(self) -> int:
        return max((d for c in self for d in c.dims), default=-1) + 1

    def permutation_forward(self) -> list[int]:
        return [d for c in self for d in c.dims]

    def __and__(self, other: "COOConditions") -> "COOConditions":
        groups: dict[int, list[COOCondition]] = {}
        groups_map: dict[int, int] = {}
        for a in list(self) + list(other):
            merge_ids = {groups_map[d] for d in a.dims if d in groups_map}
            merge = [a]
            for m in merge_ids:
                merge.extend(groups[m])
                del groups[m]
            target_id = min(merge_ids, default=len(groups))
            groups[target_id] = merge
            for m in merge:
                for d in m.dims:
                    groups_map[d] = target_id
        return COOConditions([
            COOCondition.all(*g) if len(g) > 1 else g[0]
            for g in groups.values()
        ])

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        dims_left: list[Any] = list(range(len(self)))
        for i, c in enumerate(self):
            axis = dims_left.index(i)
            tensor = c.forward(tensor, axis)
            dims_left = dims_left[:axis] + [None] * c.K + dims_left[axis + 1:]
        perm = utils.inverse_permutation(self.permutation_forward())
        return tensor.permute(*perm) if perm else tensor

    def backward(self, tensor: torch.Tensor) -> torch.Tensor:
        assert tensor.dim() == self.dim(), \
            f"tensor dim {tensor.dim()} must match conditions dim {self.dim()}"
        dims: list[int] = list(range(tensor.dim()))
        for i, c in enumerate(self):
            dims_map = [dims.index(d) for d in c.dims]
            tensor = c.backward(tensor, dims_map)
            sorted_map = sorted(dims_map)
            contiguous = (
                c.K == 0
                or sorted_map == list(range(sorted_map[0], sorted_map[0] + c.K))
            )
            insert_pos = (
                sum(1 for d in dims if dims.index(d) < sorted_map[0] and d not in c.dims)
                if contiguous and c.K > 0
                else 0
            )
            kept = [d for idx, d in enumerate(dims) if idx not in set(dims_map)]
            dims = kept[:insert_pos] + [~i] + kept[insert_pos:]
        dims = [~d for d in dims]
        for d in dims:
            assert d >= 0, "Output dims must be a subset of input dims"
        return tensor.permute(*utils.inverse_permutation(dims))

    def is_subset(self, other: "COOConditions") -> bool:
        if self.dim() != other.dim():
            return False
        for sc in self:
            match = None
            for oc in other:
                if set(oc.dims).issubset(set(sc.dims)):
                    match = oc
                    break
            if match is None:
                return False
            if not sc.is_subset(match):
                return False
        return True

    def is_empty(self) -> bool:
        return any(c.is_empty() for c in self)


@dataclass
class MultiCOOTensor:
    tensor: FactorGraphTensor
    conditions: COOConditions

    def real_numel(self) -> int:
        return sum(f.tensor.numel() for f in self.tensor.factors)

    def _dense(self) -> torch.Tensor:
        return self.conditions.forward(self.tensor.to_dense())

    def expand_to(self, conds: COOConditions) -> "MultiCOOTensor":
        """Re-express on a finer-grained ``conds`` where ``self.conditions ⊆ conds``.

        Dense-roundtrip implementation — TODO: preserve FGT factor structure.
        """
        new_compressed = conds.backward(self._dense())
        return MultiCOOTensor(FactorGraphTensor.from_dense(new_compressed), conds)

    def shrink_to(self, conds: COOConditions) -> "MultiCOOTensor":
        """Re-express on a coarser-grained ``conds`` where ``conds ⊆ self.conditions``."""
        if (
            len(self.conditions) == len(conds)
            and all(
                sc.K == 1 and tc.K == 1 and sc.dims == tc.dims
                for sc, tc in zip(self.conditions, conds)
            )
        ):
            return self._shrink_to_k1(conds)
        new_compressed = conds.backward(self._dense())
        return MultiCOOTensor(FactorGraphTensor.from_dense(new_compressed), conds)

    def _shrink_to_k1(self, conds: COOConditions) -> "MultiCOOTensor":
        """Fast path for shrink_to when both sides are all K=1, same dim order.

        For each axis, maps target-cond values back to their positions in the
        self-cond via :func:`list_index_unique`, then calls
        :meth:`FactorGraphTensor.index_select_axis` — which preserves factor
        structure for both covered and uncovered axes.
        """
        fgt = self.tensor
        for axis, (sc, tc) in enumerate(zip(self.conditions, conds)):
            self_col = sc.table.data[0]
            target_col = tc.table.data[0]
            # Identity short-circuit: same cond or both all-values of same length.
            if (
                self_col is target_col
                and sc.N == tc.N
            ):
                continue
            if self_col is None:
                self_vals = torch.arange(sc.N, dtype=torch.int64)
            else:
                self_vals = self_col.to(torch.int64)
            if target_col is None:
                target_vals = torch.arange(tc.N, dtype=torch.int64)
            else:
                target_vals = target_col.to(torch.int64)
            if (
                self_vals.shape[0] == target_vals.shape[0]
                and torch.equal(self_vals, target_vals)
            ):
                continue
            mapping = list_index_unique(
                self_vals.reshape(-1, 1), target_vals.reshape(-1, 1)
            )
            fgt = fgt.index_select_axis(axis, mapping, tc.N)
        return MultiCOOTensor(fgt, conds)

    def _relabel_and_extend(
        self,
        dim_map: dict[int, int],
        extras: list[tuple[int, int]],
    ) -> "MultiCOOTensor":
        """Relabel MCT dims via ``dim_map`` and insert all-values conds at ``extras``.

        ``extras`` is a list of ``(new_dim_label, size)`` pairs. Each inserted
        cond is K=1, all-values on the given dim — its FGT axis is uncovered
        (size ``size`` but no factor owns it), so the expansion is purely
        structural and costs nothing.
        """
        existing_conds: list[COOCondition] = []
        for c in self.conditions:
            new_cols = [dim_map[d] for d in c.table.columns]
            new_cond = COOCondition(
                _relabel_table(c.table, new_cols), list(c.output_shape)
            )
            # Transfer canonical cache: _relabel_table only renames columns
            # (data and row order unchanged), so the original's canonical view
            # relabels identically and stays canonical. We force-populate the
            # cache on ``c`` first so successive calls skip the sort.
            c_canon = c.canonical()
            if c_canon is not c:
                cached_new_cols = [dim_map[d] for d in c_canon.table.columns]
                cached_new = COOCondition(
                    _relabel_table(c_canon.table, cached_new_cols),
                    list(c_canon.output_shape),
                )
                object.__setattr__(new_cond, "_canonical_cache", cached_new)
            existing_conds.append(new_cond)
        extra_conds = [_all_values_cond(new_dim, size) for new_dim, size in extras]

        all_conds = existing_conds + extra_conds
        sources: list[int] = list(range(len(existing_conds))) + [-1] * len(extra_conds)
        idx_order = sorted(
            range(len(all_conds)), key=lambda i: tuple(all_conds[i].dims)
        )
        new_conds_list = [all_conds[i] for i in idx_order]
        new_sources = [sources[i] for i in idx_order]

        new_axis_of_old: dict[int, int] = {}
        new_shape: list[int] = []
        for new_axis, (src, cond) in enumerate(zip(new_sources, new_conds_list)):
            if src >= 0:
                new_axis_of_old[src] = new_axis
                new_shape.append(int(self.tensor.shape[src]))
            else:
                new_shape.append(int(cond.N))

        new_factors = [
            FactorTensor(f.tensor, [new_axis_of_old[d] for d in f.dims])
            for f in self.tensor.factors
        ]
        new_fgt = FactorGraphTensor(new_factors, torch.Size(new_shape))
        return MultiCOOTensor(new_fgt, COOConditions(new_conds_list))

    def clone(self) -> "MultiCOOTensor":
        new_factors = [
            FactorTensor(f.tensor.clone(), list(f.dims))
            for f in self.tensor.factors
        ]
        return MultiCOOTensor(
            FactorGraphTensor(new_factors, self.tensor.shape), self.conditions
        )

    def __add__(self, other: "MultiCOOTensor") -> Optional["MultiCOOTensor"]:
        if self.conditions is other.conditions:
            return MultiCOOTensor(self.tensor + other.tensor, self.conditions)
        if self.conditions.is_subset(other.conditions):
            return MultiCOOTensor(
                self.expand_to(other.conditions).tensor + other.tensor,
                other.conditions,
            )
        if other.conditions.is_subset(self.conditions):
            return MultiCOOTensor(
                self.tensor + other.expand_to(self.conditions).tensor,
                self.conditions,
            )
        return None

    def __iadd__(self, other: "MultiCOOTensor") -> "MultiCOOTensor":
        assert other.conditions.is_subset(self.conditions)
        self.tensor = self.tensor + other.expand_to(self.conditions).tensor
        return self

    def __isub__(self, other: "MultiCOOTensor") -> "MultiCOOTensor":
        assert other.conditions.is_subset(self.conditions)
        self.tensor = self.tensor - other.expand_to(self.conditions).tensor
        return self

    def __sub__(self, other: "MultiCOOTensor") -> Optional["MultiCOOTensor"]:
        if self.conditions is other.conditions:
            return MultiCOOTensor(self.tensor - other.tensor, self.conditions)
        if self.conditions.is_subset(other.conditions):
            return MultiCOOTensor(
                self.expand_to(other.conditions).tensor - other.tensor,
                other.conditions,
            )
        if other.conditions.is_subset(self.conditions):
            return MultiCOOTensor(
                self.tensor - other.expand_to(self.conditions).tensor,
                self.conditions,
            )
        return None

    def apply_multiplicative(
        self, fn: Callable[[torch.Tensor], torch.Tensor]
    ) -> "MultiCOOTensor":
        new_dense = fn(self.tensor.to_dense())
        return MultiCOOTensor(FactorGraphTensor.from_dense(new_dense), self.conditions)

    def apply_multiplicative_(
        self, fn: Callable[[torch.Tensor], torch.Tensor]
    ) -> None:
        self.tensor = FactorGraphTensor.from_dense(fn(self.tensor.to_dense()))

    def sum(
        self,
        dim: Union[int, list[int], None] = None,
        keepdim: bool = False,
    ) -> "MultiCOOTensor":
        """Sum over MCT output dims without materialising the dense expansion."""
        if keepdim:
            raise NotImplementedError("keepdim=True not implemented for MultiCOOTensor.sum")
        total = self.conditions.dim()
        if dim is None:
            dims_to_sum = set(range(total))
        elif isinstance(dim, int):
            dims_to_sum = {dim + total if dim < 0 else dim}
        else:
            dims_to_sum = {d + total if d < 0 else d for d in dim}

        kept_output_dims = sorted(d for d in range(total) if d not in dims_to_sum)
        remap = {d: i for i, d in enumerate(kept_output_dims)}

        plans: list[tuple[int, str, Optional[COOCondition], Optional[tuple]]] = []
        for axis, c in enumerate(self.conditions):
            summed = [d for d in c.dims if d in dims_to_sum]
            if len(summed) == c.K:
                plans.append((axis, "drop", None, None))
            elif not summed:
                relabelled = COOCondition(
                    _relabel_table(c.table, [remap[d] for d in c.table.columns]),
                    list(c.output_shape),
                )
                plans.append((axis, "keep", relabelled, None))
            else:
                new_cond, inverse = c.partial_dim_sum(summed)
                relabelled = COOCondition(
                    _relabel_table(new_cond.table, [remap[d] for d in new_cond.table.columns]),
                    list(new_cond.output_shape),
                )
                plans.append((axis, "partial", relabelled, (inverse, int(relabelled.N))))

        fgt = self.tensor
        for axis, action, _, payload in plans:
            if action == "partial":
                inverse, M = payload  # type: ignore[misc]
                if inverse is not None:
                    fgt = fgt.scatter_add_axis(axis, inverse, M)
        drop_axes = [axis for axis, action, _, _ in plans if action == "drop"]
        if drop_axes:
            fgt = fgt.sum(dim=drop_axes)

        kept = [(axis, c) for axis, action, c, _ in plans if action != "drop"]
        new_axis_of: dict[int, int] = {}
        counter = 0
        for axis, action, _, _ in plans:
            if action != "drop":
                new_axis_of[axis] = counter
                counter += 1
        kept.sort(key=lambda p: p[1].dims)
        perm = [new_axis_of[axis] for axis, _ in kept]
        if perm != list(range(len(perm))):
            fgt = fgt.permute(*perm)
        return MultiCOOTensor(fgt, COOConditions([c for _, c in kept]))

    def tensordot(
        self,
        other: "MultiCOOTensor",
        dims: Union[int, tuple[list[int], list[int]]],
    ) -> "MultiCOOTensor":
        """Tensordot on a unified dim space via shared-support intersection.

        1. Expand both operands to a unified dim layout of size
           ``len(nc_a) + len(nc_b) + k`` — non-contracted a first, then
           non-contracted b, then the k contracted pairs aligned at the tail.
           Each operand gets all-values K=1 conds added for dims it doesn't
           natively cover; these extras are structurally free (uncovered FGT
           axes).
        2. Merge conditions via ``&`` so each unified dim's support is the
           intersection of a's and b's support on that dim.
        3. Shrink both operands onto the merged conds — uses the K=1 fast
           path, which maps target-rows to self-rows via ``list_index_unique``
           and selects via ``index_select_axis`` without dense roundtrip.
        4. Elementwise-multiply the two shrunk FGTs (they share ``merged``).
        5. Sum over the contracted unified dims.
        """
        N_a = self.conditions.dim()
        N_b = other.conditions.dim()
        if isinstance(dims, int):
            assert 0 <= dims <= min(N_a, N_b)
            contract_a = list(range(N_a - dims, N_a))
            contract_b = list(range(dims))
        else:
            contract_a = list(dims[0])
            contract_b = list(dims[1])
        assert len(contract_a) == len(contract_b)
        k = len(contract_a)

        nc_a = [d for d in range(N_a) if d not in contract_a]
        nc_b = [d for d in range(N_b) if d not in contract_b]
        len_nc_a, len_nc_b = len(nc_a), len(nc_b)

        a_to_u: dict[int, int] = {}
        for i, d in enumerate(nc_a):
            a_to_u[d] = i
        for j, d in enumerate(contract_a):
            a_to_u[d] = len_nc_a + len_nc_b + j

        b_to_u: dict[int, int] = {}
        for i, d in enumerate(nc_b):
            b_to_u[d] = len_nc_a + i
        for j, d in enumerate(contract_b):
            b_to_u[d] = len_nc_a + len_nc_b + j

        def _size_of(mct: "MultiCOOTensor", d: int) -> int:
            c = next(c for c in mct.conditions if d in c.dims)
            return int(c.output_shape[c.dims.index(d)])

        a_extras = [
            (len_nc_a + i, _size_of(other, nc_b[i])) for i in range(len_nc_b)
        ]
        b_extras = [
            (i, _size_of(self, nc_a[i])) for i in range(len_nc_a)
        ]
        for j in range(k):
            a_size = _size_of(self, contract_a[j])
            b_size = _size_of(other, contract_b[j])
            assert a_size == b_size, (
                f"contracted dim size mismatch at pair "
                f"(a={contract_a[j]}, b={contract_b[j]}): {a_size} vs {b_size}"
            )

        a_expanded = self._relabel_and_extend(a_to_u, a_extras)
        b_expanded = other._relabel_and_extend(b_to_u, b_extras)

        merged = a_expanded.conditions & b_expanded.conditions

        a_shrunk = a_expanded.shrink_to(merged)
        b_shrunk = b_expanded.shrink_to(merged)

        product = MultiCOOTensor(a_shrunk.tensor * b_shrunk.tensor, merged)

        contract_unified = [len_nc_a + len_nc_b + j for j in range(k)]
        if contract_unified:
            return product.sum(dim=contract_unified)
        return product

    def gather_indices(
        self,
        indices: torch.Tensor,
        dim: int,
        output_shape: list[int],
    ) -> "MultiCOOTensor":
        """Structurally scatter MCT output dim ``dim`` into ``N`` new axes.

        Equivalent in dense form to ``COOCondition.forward`` applied along
        ``dim``. Only the condition owning ``dim`` is rewritten — the
        compressed FGT stays put.
        """
        total = self.conditions.dim()
        if dim < 0:
            dim += total
        assert 0 <= dim < total
        N = int(indices.shape[0])
        assert len(output_shape) == N

        owner_idx = next(i for i, c in enumerate(self.conditions) if dim in c.dims)
        c = self.conditions[owner_idx]
        pos = c.dims.index(dim)
        old_size = c.output_shape[pos]
        assert int(indices.shape[1]) == old_size, \
            f"indices cols ({indices.shape[1]}) must match shape[{dim}]={old_size}"

        existing_col = c.table.data[pos]
        if existing_col is None:
            existing_col = torch.arange(c.N, dtype=torch.int64)
        existing_col = existing_col.to(torch.int64)
        # Compose indices with the existing column: for each row i in the table,
        # the single value existing_col[i] splits into N new values indices[:, existing_col[i]].
        new_rows = indices[:, existing_col]  # (N, num_rows)

        shift = N - 1
        shifted = lambda d: d if d < dim else d + shift
        new_cols = (
            [shifted(d) for d in c.table.columns[:pos]]
            + list(range(dim, dim + N))
            + [shifted(d) for d in c.table.columns[pos + 1:]]
        )
        new_data = (
            list(c.table.data[:pos])
            + [new_rows[k] for k in range(N)]
            + list(c.table.data[pos + 1:])
        )
        new_output_shape = (
            list(c.output_shape[:pos])
            + list(output_shape)
            + list(c.output_shape[pos + 1:])
        )
        new_table = TorchTable(new_cols, new_data, length=c.N)
        owner_cond = COOCondition(new_table, new_output_shape)

        new_conds: list[COOCondition] = []
        for i, cc in enumerate(self.conditions):
            if i == owner_idx:
                new_conds.append(owner_cond)
            else:
                shifted_table = _relabel_table(
                    cc.table,
                    [d if d < dim else d + shift for d in cc.table.columns],
                )
                new_conds.append(COOCondition(shifted_table, list(cc.output_shape)))
        return MultiCOOTensor(self.tensor, COOConditions(new_conds))

    def scatter_indices(
        self,
        indices: torch.Tensor,
        dims_map: list[int],
    ) -> "MultiCOOTensor":
        """Structurally gather ``N`` MCT dims into one axis of length ``K``.

        Requires ``dims_map`` to exactly match the dim set of a single condition.
        The new axis lands at ``min(dims_map)`` if contiguous, else 0.
        """
        total = self.conditions.dim()
        N = len(dims_map)
        assert int(indices.shape[0]) == N
        K = int(indices.shape[1])
        normalised = [d + total if d < 0 else d for d in dims_map]
        for d in normalised:
            assert 0 <= d < total

        owner_idx = next(
            i for i, c in enumerate(self.conditions)
            if set(c.dims) == set(normalised)
        )
        c = self.conditions[owner_idx]

        # Align old table's columns to dims_map order, compute match for each new row.
        perm = [c.table.columns.index(d) for d in normalised]
        old_mat = c.table.materialize()[:, perm].to(torch.int64)  # (N_entries, N)
        targets = indices.T.to(torch.int64)  # (K, N)
        match = (targets.unsqueeze(1) == old_mat.unsqueeze(0)).all(dim=2)  # (K, N_entries)
        has_match = match.any(dim=1)
        N_entries = int(c.N)
        j_of_k = torch.where(
            has_match, match.int().argmax(dim=1), torch.tensor(N_entries)
        )

        gathered = self.tensor.index_select_axis(owner_idx, j_of_k, K)

        sorted_map = sorted(normalised)
        contiguous = sorted_map == list(range(sorted_map[0], sorted_map[0] + N))
        insert_pos = sorted_map[0] if contiguous else 0
        kept_old_dims = [d for d in range(total) if d not in set(normalised)]
        layout = kept_old_dims[:insert_pos] + [-1] + kept_old_dims[insert_pos:]
        new_dim_of = {d: layout.index(d) for d in kept_old_dims}
        new_axis_dim = insert_pos

        new_conds: list[COOCondition] = []
        for i, cc in enumerate(self.conditions):
            if i == owner_idx:
                identity_table = TorchTable([new_axis_dim], [None], length=K)
                identity_table.is_sorted = True
                identity_table.is_unique = True
                new_conds.append(COOCondition(identity_table, [K]))
            else:
                shifted_table = _relabel_table(
                    cc.table, [new_dim_of[d] for d in cc.table.columns]
                )
                new_conds.append(COOCondition(shifted_table, list(cc.output_shape)))
        return MultiCOOTensor(gathered, COOConditions(new_conds))

    def add_intersection_to(
        self, other: "MultiCOOTensor", neg: bool = False
    ) -> "MultiCOOTensor":
        """Add (or subtract when ``neg``) self's values to ``other`` at positions in both supports.

        Fast path: when both operands share a ``conditions`` instance, delegate
        to FGT add/sub. Otherwise dense-roundtrip onto other's support.
        """
        if self.conditions is other.conditions:
            other.tensor = (
                other.tensor - self.tensor if neg else other.tensor + self.tensor
            )
            return other
        self_dense = self.conditions.forward(self.tensor.to_dense())
        proj_dense = other.conditions.backward(self_dense)
        proj_fgt = FactorGraphTensor.from_dense(proj_dense)
        other.tensor = (
            other.tensor - proj_fgt if neg else other.tensor + proj_fgt
        )
        return other

    @classmethod
    def from_dense(
        cls, tensor: torch.Tensor, coo_dim_groups: list[list[int]]
    ) -> "MultiCOOTensor":
        """Compress ``tensor`` under a per-group sparse support, one condition per group."""
        covered = [d for g in coo_dim_groups for d in g]
        assert sorted(covered) == list(range(tensor.dim())), \
            "coo_dim_groups must partition range(tensor.dim())"

        nonzero = torch.nonzero(tensor)  # (M, D)
        conditions: list[COOCondition] = []
        for group in coo_dim_groups:
            if int(nonzero.shape[0]) == 0:
                data = [torch.zeros(0, dtype=torch.int64) for _ in group]
                table = TorchTable(list(group), data, length=0)
            else:
                projected = nonzero[:, group].T.contiguous().to(torch.int64)
                unique = torch.unique(projected, dim=1)  # (K, N_unique)
                data = [unique[k] for k in range(len(group))]
                table = TorchTable(list(group), data)
            table, _ = table.sort_dedup()
            output_shape = [int(tensor.shape[d]) for d in group]
            # Reorder output_shape to match sorted columns.
            group_to_size = dict(zip(group, output_shape))
            output_shape = [group_to_size[d] for d in table.columns]
            conditions.append(COOCondition(table, output_shape))
        conds = COOConditions(conditions)
        compressed = conds.backward(tensor)
        return cls(FactorGraphTensor.from_dense(compressed), conds)


@dataclass
class MultiCOOTensorSum:
    terms: list[MultiCOOTensor]

    def __post_init__(self):
        self._assert_sorted()

    def add_term(self, term: MultiCOOTensor):
        for t in range(len(self.terms)):
            if term.real_numel() > self.terms[t].real_numel():
                break
            if tensor := self.terms[t].__add__(term):
                self.terms[t] = tensor
                self._assert_sorted()
                return
        else:
            t = len(self.terms)
        rest = self.terms[t:]
        self.terms = self.terms[:t] + [term]
        for term in rest:
            if tensor := term.__add__(self.terms[t]):
                self.terms[t] = tensor
            else:
                self.terms.append(term)
        self._assert_sorted()

    def _assert_sorted(self):
        assert all(
            t.real_numel() >= self.terms[i + 1].real_numel()
            for i, t in enumerate(self.terms[:-1])
        ), "Terms must be sorted by real_numel in descending order"

    def apply_multiplicative(
        self, fn: Callable[[torch.Tensor], torch.Tensor]
    ) -> "MultiCOOTensorSum":
        old_terms = self.terms
        new_terms: list[MultiCOOTensor] = []
        for i, t1 in enumerate(old_terms):
            new_t = t1.clone()
            for t2 in old_terms[:i]:
                t2.add_intersection_to(new_t)
            new_t.apply_multiplicative_(fn)
            for t in new_terms:
                t.add_intersection_to(new_t, neg=True)
            new_terms.append(new_t)
        return MultiCOOTensorSum(new_terms)


__all__ = ["COOCondition", "COOConditions", "MultiCOOTensor", "MultiCOOTensorSum"]
