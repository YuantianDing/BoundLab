from dataclasses import dataclass
from typing import Iterable, Union

import torch

from boundlab import utils


Scalar = Union[int, float, torch.Tensor]


def _scatter_axis_via_indices(
    tensor: torch.Tensor,
    dim: int,
    indices: torch.Tensor,
    output_shape: list[int],
) -> torch.Tensor:
    """Scatter ``tensor``'s ``dim`` (length K) into ``N = indices.shape[0]`` new
    axes with sizes ``output_shape``.

    For each ``k ∈ [0, K)``, the slice ``tensor[..., k, ...]`` along ``dim`` is
    placed at output position ``indices[:, k]`` across the N new axes; all
    other output positions are zero. Assumes ``indices`` holds unique columns
    (no duplicate target positions).
    """
    if dim < 0:
        dim += tensor.ndim
    N = int(indices.shape[0])
    K = int(tensor.shape[dim])
    assert N == len(output_shape)
    assert int(indices.shape[1]) == K

    out_shape = (
        list(tensor.shape[:dim]) + list(output_shape) + list(tensor.shape[dim + 1:])
    )
    if K == 0 or N == 0:
        return torch.zeros(out_shape, dtype=tensor.dtype, device=tensor.device)

    strides = [1] * N
    for k in range(N - 2, -1, -1):
        strides[k] = strides[k + 1] * int(output_shape[k + 1])
    stride_tensor = torch.tensor(strides, dtype=torch.int64)
    lin = (indices.to(torch.int64).T * stride_tensor).sum(dim=1)  # (K,)

    prod = 1
    for s in output_shape:
        prod *= int(s)
    flat_shape = list(tensor.shape[:dim]) + [prod] + list(tensor.shape[dim + 1:])
    out_flat = torch.zeros(flat_shape, dtype=tensor.dtype, device=tensor.device)
    out_flat.index_copy_(dim, lin, tensor)
    return out_flat.reshape(out_shape)


def _gather_axes_via_indices(
    tensor: torch.Tensor,
    dims_map: list[int],
    indices: torch.Tensor,
    output_shape: list[int],
) -> torch.Tensor:
    """Reverse of ``_scatter_axis_via_indices``.

    The N axes listed in ``dims_map`` (each of size ``output_shape[i]``)
    collapse into a single axis of length ``K = indices.shape[1]``. The new
    axis lands at ``min(dims_map)`` when ``dims_map`` is contiguous, else at
    position 0 (matching PyTorch advanced-indexing semantics).
    """
    N = len(dims_map)
    K = int(indices.shape[1])
    assert int(indices.shape[0]) == N
    normalised = [d + tensor.ndim if d < 0 else d for d in dims_map]
    assert all(int(tensor.shape[normalised[k]]) == int(output_shape[k]) for k in range(N))

    non_map = [d for d in range(tensor.ndim) if d not in normalised]
    perm = list(normalised) + list(non_map)
    permuted = tensor.permute(*perm)
    trailing = list(permuted.shape[N:])
    prod = 1
    for s in output_shape:
        prod *= int(s)
    flat = permuted.reshape(prod, *trailing) if N > 0 else permuted

    if N == 0 or K == 0:
        empty_shape = [K] + trailing
        return torch.zeros(empty_shape, dtype=tensor.dtype, device=tensor.device)

    strides = [1] * N
    for k in range(N - 2, -1, -1):
        strides[k] = strides[k + 1] * int(output_shape[k + 1])
    stride_tensor = torch.tensor(strides, dtype=torch.int64)
    lin = (indices.to(torch.int64).T * stride_tensor).sum(dim=1)  # (K,)

    gathered = flat.index_select(0, lin)  # (K, *trailing)

    sorted_map = sorted(normalised)
    contiguous = sorted_map == list(range(sorted_map[0], sorted_map[0] + N))
    insert_final = (
        sum(1 for d in non_map if d < sorted_map[0]) if contiguous else 0
    )
    if insert_final == 0 or gathered.ndim == 1:
        return gathered
    perm_final = (
        list(range(1, insert_final + 1))
        + [0]
        + list(range(insert_final + 1, gathered.ndim))
    )
    return gathered.permute(*perm_final)


@dataclass
class FactorTensor:

    tensor: torch.Tensor
    dims: list[int]

    def __post_init__(self):
        assert self.tensor.ndim == len(self.dims)

    def expand_to(self, target_dims: list[int]) -> torch.Tensor:
        return _align_factor(self.tensor, self.dims, target_dims)


def _align_factor(
    tensor: torch.Tensor, source_dims: list[int], target_dims: list[int]
) -> torch.Tensor:
    """Reshape ``tensor`` (labelled by ``source_dims``) into ``target_dims`` layout.

    Source dims must be a subset of target dims; axes for missing target labels
    are inserted as size-1 dims so the returned tensor broadcasts against a
    tensor with ``target_dims`` layout.
    """
    assert set(source_dims).issubset(set(target_dims))
    ordered = [d for d in target_dims if d in source_dims]
    perm = [source_dims.index(d) for d in ordered]
    out = tensor.permute(*perm) if perm else tensor
    for i, d in enumerate(target_dims):
        if d not in source_dims:
            out = out.unsqueeze(i)
    return out


def _canonicalize(factors: list[FactorTensor]) -> list[FactorTensor]:
    """Fuse factors with identical dim sets and fold proper-subset factors into supersets."""
    factors = list(factors)
    # Fuse identical-dim-set factors.
    i = 0
    while i < len(factors):
        j = i + 1
        while j < len(factors):
            if set(factors[i].dims) == set(factors[j].dims):
                aligned = _align_factor(factors[j].tensor, factors[j].dims, factors[i].dims)
                factors[i] = FactorTensor(factors[i].tensor * aligned, factors[i].dims)
                del factors[j]
            else:
                j += 1
        i += 1
    # Fold proper-subset factors into one of their supersets.
    changed = True
    while changed:
        changed = False
        for i in range(len(factors)):
            absorbed = False
            for j in range(len(factors)):
                if i == j:
                    continue
                if set(factors[i].dims) < set(factors[j].dims):
                    aligned = _align_factor(factors[i].tensor, factors[i].dims, factors[j].dims)
                    factors[j] = FactorTensor(factors[j].tensor * aligned, factors[j].dims)
                    del factors[i]
                    absorbed = True
                    changed = True
                    break
            if absorbed:
                break
    return factors


@dataclass
class FactorGraphTensor:
    """A tensor represented as a mul of factors.

    Represents ``T[d_0, ..., d_{n-1}] = Π_i factor_i[*factor_i.dims]`` where
    dims not referenced by any factor are broadcast (constant along that axis).
    The dense ``shape`` is stored explicitly so uncovered dims still have a
    well-defined size.

    Canonical-form invariants:
      * No two factors share the same dim-set.
      * No factor's dim-set is a proper subset of another's.
    """

    factors: list[FactorTensor]
    shape: torch.Size

    def working_dimensions(self) -> list[int]:
        result = set()
        for f in self.factors:
            result.update(f.dims)
        return [a for a in sorted(result) if self.shape[a] > 1]
        


    def __post_init__(self):
        if not isinstance(self.shape, torch.Size):
            self.shape = torch.Size(self.shape)
        for i, factor in enumerate(self.factors):
            for local_d, out_d in enumerate(factor.dims):
                assert 0 <= out_d < len(self.shape), \
                    f"Factor dim {out_d} out of range for shape {tuple(self.shape)}"
                assert factor.tensor.shape[local_d] == self.shape[out_d], \
                    f"Factor dim {out_d} size {factor.tensor.shape[local_d]} ≠ shape[{out_d}]={self.shape[out_d]}"
                assert factor.tensor.shape[local_d] > 1, \
                    f"Factor has size-1 dim at local axis {local_d} (output dim {out_d}); " \
                    f"size-1 dims must be left uncovered"
                assert factor.tensor.stride(local_d) != 0, \
                    f"Factor has expanded (stride-0) dim at local axis {local_d} (output dim {out_d}); " \
                    f"broadcast dims must be left uncovered"
            for factor2 in self.factors[i + 1:]:
                assert not set(factor.dims).issubset(set(factor2.dims)), \
                    f"Factors not in canonical form: {factor.dims} ⊆ {factor2.dims}"
                assert not set(factor2.dims).issubset(set(factor.dims)), \
                    f"Factors not in canonical form: {factor2.dims} ⊆ {factor.dims}"

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def dim(self) -> int:
        return len(self.shape)

    def numel(self) -> int:
        return self.shape.numel()

    @property
    def covered_dims(self) -> list[int]:
        return sorted({d for f in self.factors for d in f.dims})

    @property
    def uncovered_dims(self) -> list[int]:
        covered = {d for f in self.factors for d in f.dims}
        return [d for d in range(self.ndim) if d not in covered]

    def cover_dims(self, extra_dims: Iterable[int]) -> "FactorGraphTensor":
        """Return an equivalent FGT whose factors collectively cover
        ``covered_dims ∪ extra_dims`` (restricted to size>1 dims — size-1
        dims must stay uncovered to satisfy the invariant). Newly-covered
        dims are materialised (broadcast-then-contiguous); all existing
        factors fuse into a single factor on the union."""
        covered = set(self.covered_dims)
        extra = [d for d in extra_dims if d not in covered and self.shape[d] > 1]
        if not extra:
            return self
        fused_tensor, fused_dims = self._fuse_all()
        union = sorted(set(fused_dims) | set(extra))
        aligned = _align_factor(fused_tensor, fused_dims, union)
        target_shape = [int(self.shape[d]) for d in union]
        materialised = aligned.expand(target_shape).contiguous()
        return FactorGraphTensor([FactorTensor(materialised, union)], self.shape)

    @classmethod
    def from_dense(cls, tensor: torch.Tensor) -> "FactorGraphTensor":
        """Wrap a dense ``tensor`` as a single-factor ``FactorGraphTensor``.

        Size-1 dims are left uncovered (no factor may own a size-1 axis); the
        factor tensor holds only the size>1 axes and is made contiguous so
        stride-0 broadcast axes are materialised. Empty tensors (numel==0)
        produce a factor-less FGT.
        """
        shape = tensor.shape
        if tensor.numel() == 0:
            return cls([], shape)
        dims = [d for d in range(tensor.ndim) if shape[d] > 1]
        factor_tensor = tensor.contiguous().reshape([shape[d] for d in dims])
        return cls([FactorTensor(factor_tensor, dims)], shape)

    def to_dense(self) -> torch.Tensor:
        """Convert the factor graph tensor to a dense tensor.

        Dims not referenced by any factor contribute a ``torch.ones(size)``
        template to the einsum so the result has the declared shape.
        """
        covered: set[int] = set()
        for f in self.factors:
            covered.update(f.dims)
        operands: list = []
        for f in self.factors:
            operands.extend([f.tensor, list(f.dims)])
        for d in range(self.ndim):
            if d not in covered:
                operands.extend([torch.ones(self.shape[d]), [d]])
        if not operands:
            return torch.tensor(1.0)
        operands.append(list(range(self.ndim)))
        return torch.einsum(*operands)

    # ------------------------------------------------------------------
    # Elementwise arithmetic
    # ------------------------------------------------------------------
    def _scale(self, value: Scalar) -> "FactorGraphTensor":
        if not self.factors:
            return FactorGraphTensor(
                [FactorTensor(torch.as_tensor(value, dtype=torch.get_default_dtype()).reshape(()), [])],
                self.shape,
            )
        new_factors = list(self.factors)
        f0 = new_factors[0]
        new_factors[0] = FactorTensor(f0.tensor * value, list(f0.dims))
        return FactorGraphTensor(new_factors, self.shape)

    def __mul__(self, other) -> "FactorGraphTensor":
        if isinstance(other, FactorGraphTensor):
            assert tuple(self.shape) == tuple(other.shape), \
                f"Shape mismatch: {tuple(self.shape)} vs {tuple(other.shape)}"
            merged = _canonicalize(list(self.factors) + list(other.factors))
            return FactorGraphTensor(merged, self.shape)
        if isinstance(other, torch.Tensor):
            if other.ndim != 0:
                return NotImplemented
            return self._scale(other)
        if isinstance(other, (int, float)):
            return self._scale(other)
        return NotImplemented

    def __rmul__(self, other) -> "FactorGraphTensor":
        return self.__mul__(other)

    def _fuse_all(self) -> tuple[torch.Tensor, list[int]]:
        """Fuse all factors via einsum into a single tensor on sorted(covered).

        Returns ``(tensor, covered_dims)``; an FGT with no factors yields
        ``(tensor(1.0), [])``.
        """
        covered = sorted({d for f in self.factors for d in f.dims})
        if not self.factors:
            return torch.tensor(1.0), []
        operands: list = []
        for f in self.factors:
            operands.extend([f.tensor, list(f.dims)])
        operands.append(list(covered))
        return torch.einsum(*operands), covered

    def __add__(self, other) -> "FactorGraphTensor":
        # Lift scalars into a 0-dim constant FGT so the union logic below
        # handles them uniformly.
        if isinstance(other, (int, float)):
            other = FactorGraphTensor(
                [FactorTensor(torch.tensor(float(other)), [])], self.shape
            )
        elif isinstance(other, torch.Tensor):
            if other.ndim != 0:
                return NotImplemented
            other = FactorGraphTensor(
                [FactorTensor(other.reshape(()), [])], self.shape
            )
        if not isinstance(other, FactorGraphTensor):
            return NotImplemented
        assert tuple(self.shape) == tuple(other.shape), \
            f"Shape mismatch: {tuple(self.shape)} vs {tuple(other.shape)}"

        a_tensor, a_dims = self._fuse_all()
        b_tensor, b_dims = other._fuse_all()
        union = sorted(set(a_dims) | set(b_dims))

        if not union:
            # Both sides are scalar-constant FGTs — sum folds into a single
            # 0-dim factor (no covered dims).
            return FactorGraphTensor(
                [FactorTensor((a_tensor + b_tensor).reshape(()), [])],
                self.shape,
            )

        # Broadcast each fused tensor into the union's layout; addition
        # materialises the result to the max shape, so the resulting factor
        # satisfies the size>1 / stride!=0 invariant (union dims are always
        # covered by at least one side, hence shape[d] > 1).
        a_aligned = _align_factor(a_tensor, a_dims, union)
        b_aligned = _align_factor(b_tensor, b_dims, union)
        return FactorGraphTensor(
            [FactorTensor(a_aligned + b_aligned, union)], self.shape,
        )

    def __radd__(self, other) -> "FactorGraphTensor":
        return self.__add__(other)

    def __sub__(self, other) -> "FactorGraphTensor":
        if isinstance(other, FactorGraphTensor):
            return self.__add__(-other)
        if isinstance(other, (int, float)):
            return self.__add__(-other)
        if isinstance(other, torch.Tensor):
            if other.ndim != 0:
                return NotImplemented
            return self.__add__(-other)
        return NotImplemented

    def __rsub__(self, other) -> "FactorGraphTensor":
        return (-self).__add__(other)

    def __truediv__(self, other) -> "FactorGraphTensor":
        if isinstance(other, torch.Tensor):
            if other.ndim != 0:
                return NotImplemented
            return self._scale(1.0 / other)
        if isinstance(other, (int, float)):
            return self._scale(1.0 / other)
        return NotImplemented

    def __neg__(self) -> "FactorGraphTensor":
        return self._scale(-1)

    def __pos__(self) -> "FactorGraphTensor":
        return self

    # ------------------------------------------------------------------
    # Shape-only transforms
    # ------------------------------------------------------------------
    def permute(self, *dims) -> "FactorGraphTensor":
        if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
            dims = tuple(dims[0])
        assert sorted(dims) == list(range(self.ndim)), \
            f"Invalid permutation {dims} for ndim={self.ndim}"
        inv = utils.inverse_permutation(list(dims))
        new_factors = [
            FactorTensor(f.tensor, [inv[d] for d in f.dims]) for f in self.factors
        ]
        new_shape = torch.Size([self.shape[d] for d in dims])
        return FactorGraphTensor(new_factors, new_shape)

    def transpose(self, dim0: int, dim1: int) -> "FactorGraphTensor":
        if dim0 < 0:
            dim0 += self.ndim
        if dim1 < 0:
            dim1 += self.ndim
        perm = list(range(self.ndim))
        perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
        return self.permute(*perm)

    def unsqueeze(self, dim: int) -> "FactorGraphTensor":
        if dim < 0:
            dim += self.ndim + 1
        assert 0 <= dim <= self.ndim, f"Invalid unsqueeze dim: {dim}"
        shifted = [
            FactorTensor(f.tensor, [d if d < dim else d + 1 for d in f.dims])
            for f in self.factors
        ]
        new_shape = torch.Size(list(self.shape)[:dim] + [1] + list(self.shape)[dim:])
        return FactorGraphTensor(shifted, new_shape)

    def squeeze(self, dim: int | None = None) -> "FactorGraphTensor":
        if dim is None:
            result = self
            d = result.ndim - 1
            while d >= 0:
                if result.shape[d] == 1:
                    result = result._squeeze_one(d)
                d -= 1
            return result
        if dim < 0:
            dim += self.ndim
        assert 0 <= dim < self.ndim, f"Invalid squeeze dim: {dim}"
        if self.shape[dim] != 1:
            return self
        return self._squeeze_one(dim)

    def _squeeze_one(self, dim: int) -> "FactorGraphTensor":
        # Under the size>1 invariant, no factor can own ``dim`` (which is size 1),
        # so squeezing only re-labels the higher output dims of each factor.
        assert all(dim not in f.dims for f in self.factors)
        new_factors = [
            FactorTensor(f.tensor, [d if d < dim else d - 1 for d in f.dims])
            for f in self.factors
        ]
        new_shape = torch.Size(list(self.shape)[:dim] + list(self.shape)[dim + 1:])
        return FactorGraphTensor(new_factors, new_shape)

    def expand(self, *sizes) -> "FactorGraphTensor":
        # Under the size>1 invariant, only size-1 (always uncovered) dims are
        # expandable, so ``expand`` only needs to update the declared shape;
        # the factor list is reused unchanged.
        if len(sizes) == 1 and isinstance(sizes[0], (tuple, list, torch.Size)):
            sizes = tuple(sizes[0])
        assert len(sizes) == self.ndim, \
            f"expand size mismatch: got {len(sizes)}, expected {self.ndim}"
        new_shape = list(self.shape)
        for d, new_size in enumerate(sizes):
            if new_size == -1 or new_size == self.shape[d]:
                continue
            assert self.shape[d] == 1, \
                f"expand: dim {d} has size {self.shape[d]} ≠ 1, cannot expand to {new_size}"
            new_shape[d] = new_size
        return FactorGraphTensor(list(self.factors), torch.Size(new_shape))

    # ------------------------------------------------------------------
    # Reductions
    # ------------------------------------------------------------------
    def sum(
        self,
        dim: Union[int, Iterable[int], None] = None,
        keepdim: bool = False,
    ) -> "FactorGraphTensor":
        """Sum along ``dim``.

        A dim owned by at most one factor sums cheaply; uncovered dims reduce to
        a scalar multiplier of that dim's size. Summing a dim shared across
        multiple factors fuses those factors with ``torch.einsum`` first so the
        dim ends up with a single owner, then sums it locally.
        """
        if dim is None:
            dims = list(range(self.ndim))
        elif isinstance(dim, int):
            dims = [dim]
        else:
            dims = list(dim)
        normalised = sorted(
            {d + self.ndim if d < 0 else d for d in dims}, reverse=True
        )
        result = self
        for d in normalised:
            result = result._sum_one(d, keepdim)
        return result
    
    def gather_indices(
        self,
        indices: torch.Tensor,
        dim: int,
        output_shape: list[int],
    ) -> "FactorGraphTensor":
        """Scatter axis ``dim`` (length ``K``) into ``N = indices.shape[0]`` new
        axes of sizes ``output_shape``: for each ``k``, the value at input
        position ``k`` along ``dim`` is placed at output position
        ``indices[:, k]`` across the N new axes; all other positions are zero.
        """
        # TODO: preserve factor structure instead of dense roundtrip.
        if dim < 0:
            dim += self.ndim
        assert 0 <= dim < self.ndim
        N = int(indices.shape[0])
        K = int(self.shape[dim])
        assert N == len(output_shape), "indices rows must match output_shape length"
        assert int(indices.shape[1]) == K, \
            f"indices cols ({indices.shape[1]}) must match shape[{dim}]={K}"
        dense = self.to_dense()
        out = _scatter_axis_via_indices(dense, dim, indices, list(output_shape))
        return FactorGraphTensor.from_dense(out)

    def scatter_indices(
        self,
        indices: torch.Tensor,
        dims_map: list[int],
    ) -> "FactorGraphTensor":
        """``dims_map[i]`` is the FactorGraphTensor dim selected by row ``i`` of
        ``indices``.  The ``N = indices.shape[0]`` indexed axes collapse into
        a single axis of length ``K = indices.shape[1]``; that axis lands at
        ``min(dims_map)`` when ``dims_map`` is contiguous, and at position 0
        otherwise, matching PyTorch advanced-indexing semantics.
        """
        # TODO: preserve factor structure instead of dense roundtrip.
        assert int(indices.shape[0]) == len(dims_map), \
            "indices rows must match dims_map length"
        normalised = [d + self.ndim if d < 0 else d for d in dims_map]
        for d in normalised:
            assert 0 <= d < self.ndim
        output_shape = [int(self.shape[d]) for d in normalised]
        dense = self.to_dense()
        out = _gather_axes_via_indices(dense, normalised, indices, output_shape)
        return FactorGraphTensor.from_dense(out)

    def _sum_one(self, d: int, keepdim: bool) -> "FactorGraphTensor":
        owners = [i for i, f in enumerate(self.factors) if d in f.dims]
        if len(owners) > 1:
            # Fuse the owner factors via einsum so dim d has a single owner,
            # then recurse to take the usual local sum.
            union = sorted({dd for i in owners for dd in self.factors[i].dims})
            operands: list = []
            for i in owners:
                f = self.factors[i]
                operands.extend([f.tensor, list(f.dims)])
            operands.append(list(union))
            fused = torch.einsum(*operands)
            new_factors: list[FactorTensor] = []
            for i, ft in enumerate(self.factors):
                if i == owners[0]:
                    new_factors.append(FactorTensor(fused, list(union)))
                elif i in owners:
                    continue
                else:
                    new_factors.append(FactorTensor(ft.tensor, list(ft.dims)))
            fused_fgt = FactorGraphTensor(_canonicalize(new_factors), self.shape)
            return fused_fgt._sum_one(d, keepdim)
        size = int(self.shape[d])
        if keepdim:
            # Summing over dim d collapses it to size 1; under the size>1 invariant
            # no factor may cover a size-1 dim, so we drop d from the owner's dim
            # list (sum without keepdim) and leave the output shape with 1 at d.
            new_factors: list[FactorTensor] = []
            if owners:
                idx = owners[0]
                f = self.factors[idx]
                local = f.dims.index(d)
                new_tensor = f.tensor.sum(dim=local)
                new_dims_for_f = [dd for dd in f.dims if dd != d]
                for i, ft in enumerate(self.factors):
                    if i == idx:
                        new_factors.append(FactorTensor(new_tensor, new_dims_for_f))
                    else:
                        new_factors.append(FactorTensor(ft.tensor, list(ft.dims)))
            else:
                new_factors = [
                    FactorTensor(ft.tensor, list(ft.dims)) for ft in self.factors
                ]
            new_shape = torch.Size(list(self.shape)[:d] + [1] + list(self.shape)[d + 1:])
            result = FactorGraphTensor(_canonicalize(new_factors), new_shape)
            if not owners:
                result = result._scale(size)
            return result
        # keepdim=False: drop dim d entirely and shift higher dims down by one.
        new_factors: list[FactorTensor] = []
        if owners:
            idx = owners[0]
            f = self.factors[idx]
            local = f.dims.index(d)
            new_tensor = f.tensor.sum(dim=local)
            new_dims_for_f = [dd if dd < d else dd - 1 for dd in f.dims if dd != d]
            for i, ft in enumerate(self.factors):
                if i == idx:
                    new_factors.append(FactorTensor(new_tensor, new_dims_for_f))
                else:
                    new_factors.append(
                        FactorTensor(ft.tensor, [dd if dd < d else dd - 1 for dd in ft.dims])
                    )
        else:
            for ft in self.factors:
                new_factors.append(
                    FactorTensor(ft.tensor, [dd if dd < d else dd - 1 for dd in ft.dims])
                )
        new_factors = _canonicalize(new_factors)
        new_shape = torch.Size(list(self.shape)[:d] + list(self.shape)[d + 1:])
        result = FactorGraphTensor(new_factors, new_shape)
        if not owners:
            result = result._scale(size)
        return result

    def mean(
        self,
        dim: Union[int, Iterable[int], None] = None,
        keepdim: bool = False,
    ) -> "FactorGraphTensor":
        if dim is None:
            dims = list(range(self.ndim))
        elif isinstance(dim, int):
            dims = [dim]
        else:
            dims = list(dim)
        normalised = {d + self.ndim if d < 0 else d for d in dims}
        divisor = 1
        for d in normalised:
            divisor *= int(self.shape[d])
        return self.sum(dim=dim, keepdim=keepdim) / divisor

    # ------------------------------------------------------------------
    # Structured axis rewrites (scatter-add / index-select / diagonal)
    # ------------------------------------------------------------------
    def _fuse_owners(self, axis: int) -> tuple["FactorGraphTensor", int]:
        """If ``axis`` has more than one owner factor, fuse them into one via einsum.

        Returns ``(fgt, owner_idx)`` where ``owner_idx`` is the index of the
        single owner factor (or ``-1`` when no factor covers ``axis``).
        """
        owners = [i for i, f in enumerate(self.factors) if axis in f.dims]
        if len(owners) <= 1:
            return self, owners[0] if owners else -1
        union = sorted({d for i in owners for d in self.factors[i].dims})
        operands: list = []
        for i in owners:
            f = self.factors[i]
            operands.extend([f.tensor, list(f.dims)])
        operands.append(list(union))
        fused_tensor = torch.einsum(*operands)
        first = owners[0]
        new_factors: list[FactorTensor] = []
        for i, ft in enumerate(self.factors):
            if i == first:
                new_factors.append(FactorTensor(fused_tensor, union))
            elif i in owners:
                continue
            else:
                new_factors.append(FactorTensor(ft.tensor, list(ft.dims)))
        new_fgt = FactorGraphTensor(new_factors, self.shape)
        new_owner = next(
            i for i, f in enumerate(new_fgt.factors) if axis in f.dims
        )
        return new_fgt, new_owner

    def scatter_add_axis(
        self, axis: int, inverse: torch.Tensor, M: int
    ) -> "FactorGraphTensor":
        """Replace ``axis`` (size ``N``) with a new axis of length ``M`` via
        ``index_add_`` using ``inverse`` (values in ``[0, M)``, length ``N``).

        Preserves factor structure: when ``axis`` has a single owner factor,
        only that factor's local axis is scattered; uncovered ``axis`` reduces
        to a bincount-weighted broadcast (scaled in the ``M == 1`` degenerate
        case). Multi-owner ``axis`` is first fused via ``_fuse_owners``.
        """
        N = int(self.shape[axis])
        assert inverse.shape == torch.Size([N]), \
            f"inverse shape {tuple(inverse.shape)} must be ({N},)"
        inverse = inverse.to(torch.int64)
        fgt, owner = self._fuse_owners(axis)

        new_shape = torch.Size(
            list(fgt.shape)[:axis] + [M] + list(fgt.shape)[axis + 1:]
        )

        if owner >= 0:
            f = fgt.factors[owner]
            local = f.dims.index(axis)
            new_local_shape = list(f.tensor.shape)
            new_local_shape[local] = M
            new_t = torch.zeros(
                new_local_shape, dtype=f.tensor.dtype, device=f.tensor.device
            )
            new_t.index_add_(local, inverse, f.tensor)
            result_factors: list[FactorTensor] = []
            if M > 1:
                result_factors.append(FactorTensor(new_t, list(f.dims)))
            else:
                result_factors.append(
                    FactorTensor(
                        new_t.squeeze(local),
                        [d for d in f.dims if d != axis],
                    )
                )
            for i, ft in enumerate(fgt.factors):
                if i != owner:
                    result_factors.append(FactorTensor(ft.tensor, list(ft.dims)))
            return FactorGraphTensor(_canonicalize(result_factors), new_shape)

        counts = torch.bincount(inverse, minlength=M)
        if M > 1:
            result_factors = [
                FactorTensor(ft.tensor, list(ft.dims)) for ft in fgt.factors
            ]
            result_factors.append(FactorTensor(counts, [axis]))
            return FactorGraphTensor(_canonicalize(result_factors), new_shape)
        # M == 1: axis stays uncovered (size 1); scale by the single bincount value.
        scale = counts[0]
        if fgt.factors:
            result_factors = [
                FactorTensor(ft.tensor, list(ft.dims)) for ft in fgt.factors
            ]
            f0 = result_factors[0]
            result_factors[0] = FactorTensor(f0.tensor * scale, list(f0.dims))
        else:
            result_factors = [FactorTensor(scale.reshape(()), [])]
        return FactorGraphTensor(_canonicalize(result_factors), new_shape)

    def index_select_axis(
        self, axis: int, index: torch.Tensor, K: int
    ) -> "FactorGraphTensor":
        """Replace ``axis`` (size ``N``) with a new axis of length ``K`` via
        ``torch.index_select`` using ``index`` (values in ``[0, N]``; the
        sentinel ``N`` produces a zero row).

        The zero sentinel is materialised by padding the owner factor with an
        extra zero slice along its local axis for ``axis``. Uncovered ``axis``
        is replaced by a new 1-D factor ``ones_with_zero_sentinel[index]``.
        """
        N = int(self.shape[axis])
        index = index.to(torch.int64)
        fgt, owner = self._fuse_owners(axis)

        new_shape = torch.Size(
            list(fgt.shape)[:axis] + [K] + list(fgt.shape)[axis + 1:]
        )

        if owner >= 0:
            f = fgt.factors[owner]
            local = f.dims.index(axis)
            pad_shape = list(f.tensor.shape)
            pad_shape[local] = 1
            pad = torch.zeros(
                pad_shape, dtype=f.tensor.dtype, device=f.tensor.device
            )
            padded = torch.cat([f.tensor, pad], dim=local)
            new_t = torch.index_select(padded, local, index)
            result_factors: list[FactorTensor] = []
            if K > 1:
                result_factors.append(FactorTensor(new_t, list(f.dims)))
            else:
                result_factors.append(
                    FactorTensor(
                        new_t.squeeze(local),
                        [d for d in f.dims if d != axis],
                    )
                )
            for i, ft in enumerate(fgt.factors):
                if i != owner:
                    result_factors.append(FactorTensor(ft.tensor, list(ft.dims)))
            return FactorGraphTensor(_canonicalize(result_factors), new_shape)

        picker = torch.cat([
            torch.ones(N, dtype=torch.get_default_dtype()),
            torch.zeros(1, dtype=torch.get_default_dtype()),
        ])
        selected = picker[index]
        if K > 1:
            result_factors = [
                FactorTensor(ft.tensor, list(ft.dims)) for ft in fgt.factors
            ]
            result_factors.append(FactorTensor(selected, [axis]))
            return FactorGraphTensor(_canonicalize(result_factors), new_shape)
        scale = selected[0]
        if fgt.factors:
            result_factors = [
                FactorTensor(ft.tensor, list(ft.dims)) for ft in fgt.factors
            ]
            f0 = result_factors[0]
            result_factors[0] = FactorTensor(f0.tensor * scale, list(f0.dims))
        else:
            result_factors = [FactorTensor(scale.reshape(()), [])]
        return FactorGraphTensor(_canonicalize(result_factors), new_shape)

    def diagonal_independent(self, ax1: int, ax2: int) -> "FactorGraphTensor":
        """Return the diagonal of ``self`` on axes ``ax1`` and ``ax2``.

        Requires no single factor to own both axes. The lower axis is kept as
        the merged diagonal axis; the higher axis is dropped and dims above
        it shift down by one. ``_canonicalize`` fuses factors that share dim
        sets after the relabel.
        """
        assert ax1 != ax2
        lo, hi = (ax1, ax2) if ax1 < ax2 else (ax2, ax1)
        assert self.shape[lo] == self.shape[hi], \
            f"diagonal axes have mismatched sizes {self.shape[lo]} vs {self.shape[hi]}"
        for f in self.factors:
            assert not (lo in f.dims and hi in f.dims), \
                "diagonal_independent requires no factor to own both axes"
        new_shape = torch.Size(list(self.shape)[:hi] + list(self.shape)[hi + 1:])
        new_factors: list[FactorTensor] = []
        for f in self.factors:
            new_dims: list[int] = []
            for d in f.dims:
                if d == hi:
                    new_dims.append(lo)
                elif d > hi:
                    new_dims.append(d - 1)
                else:
                    new_dims.append(d)
            new_factors.append(FactorTensor(f.tensor, new_dims))
        return FactorGraphTensor(_canonicalize(new_factors), new_shape)


__all__ = ["FactorTensor", "FactorGraphTensor"]
