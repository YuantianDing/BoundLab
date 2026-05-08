
from dataclasses import dataclass
from functools import reduce
import operator
from typing import Any, Callable, Union

import torch

from boundlab.sparse.table import Indices
from boundlab.sparse.dim import Dim

        

@dataclass
class Dense:
    tensor: torch.Tensor
    dims: list[Dim]
    keep_1_dims: bool = True

    def __post_init__(self):
        assert self.tensor.ndim == len(self.dims), f"Tensor has {self.tensor.ndim} dimensions, but {len(self.dims)} dims were provided."
        assert len(set(self.dims)) == len(self.dims), "Dense tensors cannot repeat dims."
        for idx, dim in enumerate(self.dims):
            assert self.tensor.shape[idx] == dim.length, \
                f"Tensor axis {idx} has length {self.tensor.shape[idx]}, but dim has length {dim.length}."
        drops = [not self.keep_1_dims and self.tensor.shape[i] == 1 or self.tensor.stride(i) == 0 and self.tensor.shape[i] > 0 for i in range(self.tensor.ndim)]
        if True in drops:
            self.tensor = self.tensor[tuple(slice(None) if not drop else 0 for drop in drops)]
            self.dims = [dim for dim, drop in zip(self.dims, drops) if not drop]

        # TODO: sort the dims and permute the tensor accordingly, so that the dims are always in sorted order.
        sorted_dims = list(sorted(self.dims))
        if self.dims != sorted_dims:
            perm = [self.dims.index(dim) for dim in sorted_dims]
            self.tensor = self.tensor.permute(perm)
            self.dims = sorted_dims
        assert self.tensor.dim() == len(self.dims)

    @staticmethod
    def new_from(tensor: Union[torch.Tensor, "Dense", int, float]) -> "Dense":
        if isinstance(tensor, (int, float)):
            tensor = torch.tensor(tensor)
        if isinstance(tensor, Dense):
            tensor = tensor.tensor
        dims = [Dim(length=s, ordering=i) for i, s in enumerate(tensor.shape)]
        return Dense(tensor=tensor, dims=dims)

    def align_dims(self, other: "Dense") -> "Dense":
        assert len(self.dims) == len(other.dims), f"Cannot align Dense with dims {self.dims} to Dense with dims {other.dims} because they have different sets of dims."
        assert set(self.dims).isdisjoint(set(other.dims)), f"Cannot align Dense with dims {self.dims} to Dense with dims {other.dims} because they have different sets of dims."
        return Dense(tensor=other.tensor, dims=list(self.dims))

    def expand(self, dims: list[Dim]) -> torch.Tensor:
        assert set(self.dims).issubset(set(dims)), f"Cannot expand factor with dims {self.dims} to dims {dims}."
        # TODO: unsqueeze and expand and permute the tensor to match the new dims and shape.
        assert len(set(dims)) == len(dims), "Cannot expand to repeated dims."
        for dim in self.dims:
            target_dim = dims[dims.index(dim)]
            assert dim.length == target_dim.length

        tensor = self.tensor
        aligned_dims = list(self.dims)
        for idx, dim in enumerate(dims):
            if dim not in aligned_dims:
                tensor = tensor.unsqueeze(idx)
                aligned_dims.insert(idx, dim)
        perm = [aligned_dims.index(dim) for dim in dims]
        tensor = tensor.permute(perm)
        return tensor.expand([dim.length for dim in dims])
    
    def clone(self) -> "Dense":
        return Dense(tensor=self.tensor.clone(), dims=list(self.dims))

    def index_reduce_sum(
        self,
        dim: Dim,
        index: Indices,
        target_dim: Dim,
    ) -> "Dense":
        assert dim in self.dims
        idx = self.dims.index(dim)
        shape = list(self.tensor.shape)
        shape[idx] = target_dim.length
        assert len(shape) == len(self.dims)
        result = torch.zeros(shape, dtype=self.tensor.dtype, device=self.tensor.device)
        result.index_add_(idx, index, self.tensor)
        dims = list(self.dims)
        dims[idx] = target_dim
        assert result.ndim == len(self.dims)
        return Dense(tensor=result, dims=dims)

    @staticmethod
    def from_number(tensor: Union[int, float]) -> "Dense":
        return Dense(tensor=torch.tensor(tensor), dims=[])
    
    def __add__(self, other: Union["Dense", int, float]) -> "Dense":
        if not isinstance(other, Dense):
            other = Dense.from_number(other)
        dims = list(sorted(set(self.dims) | set(other.dims)))
        return Dense(tensor=self.expand(dims) + other.expand(dims), dims=dims)
    
    def __mul__(self, other: Union["Dense", int, float]) -> "Dense":
        if isinstance(other, int) and other == 1:
            return self
        if not isinstance(other, Dense):
            other = Dense.from_number(other)
        dims = list(sorted(set(self.dims) | set(other.dims)))
        return Dense(tensor=self.expand(dims) * other.expand(dims), dims=dims)
    
    def __sub__(self, other: Union["Dense", int, float]) -> "Dense":
        if not isinstance(other, Dense):
            other = Dense.from_number(other)
        dims = list(sorted(set(self.dims) | set(other.dims)))
        return Dense(tensor=self.expand(dims) - other.expand(dims), dims=dims)
    
    def __rsub__(self, other: Union["Dense", int, float]) -> "Dense":
        if not isinstance(other, Dense):
            other = Dense.from_number(other)
        dims = list(sorted(set(self.dims) | set(other.dims)))
        return Dense(tensor=other.expand(dims) - self.expand(dims), dims=dims)
    
    def __truediv__(self, other: Union["Dense", int, float]) -> "Dense":
        if not isinstance(other, Dense):
            other = Dense.from_number(other)
        dims = list(sorted(set(self.dims) | set(other.dims)))
        return Dense(tensor=self.expand(dims) / other.expand(dims), dims=dims)
    
    def __neg__(self) -> "Dense":
        return Dense(tensor=-self.tensor, dims=self.dims)
    
    def sum(self, dims: list[Dim]) -> "Dense":
        product_of_dims = reduce(operator.mul, [d.length for d in dims if d not in self.dims], 1.0)
        dims_idx = [self.dims.index(dim) for dim in dims if dim in self.dims]
        if len(dims_idx) > 0:
            tensor = self.tensor.sum(dim=dims_idx, keepdim=False)
        else:
            tensor = self.tensor
        dims = [dim for dim in self.dims if dim not in dims]
        return Dense(tensor=tensor * product_of_dims, dims=dims)
    
    def allclose(self, other: "Dense", eps: float = 1e-5, equal_nan=True, align_dims: bool = True) -> bool:
        if align_dims and set(self.dims) != set(other.dims):
            other = self.align_dims(other)
        assert set(self.dims) == set(other.dims)
        return torch.allclose(self.tensor, other.tensor, atol=eps, equal_nan=equal_nan)
    
    def diagonal(self, from_dims: list[Dim], to_dim: Dim) -> "Dense":
        assert all(dim in self.dims for dim in from_dims), f"Cannot take diagonal on dims {from_dims} that are not all in {self.dims}."
        assert to_dim not in self.dims, f"Cannot take diagonal to dim {to_dim} that is already in {self.dims}."
        assert len(from_dims) > 0, "Must provide at least one dim to diagonal."
        assert len(set(from_dims)) == len(from_dims), "Cannot diagonal repeated dims."
        assert all(dim.length == to_dim.length for dim in from_dims), \
            f"All from_dims must have the same length as to_dim {to_dim}."

        if len(from_dims) == 1:
            dims = [to_dim if dim is from_dims[0] else dim for dim in self.dims]
            return Dense(tensor=self.tensor, dims=dims)

        other_dims = [dim for dim in self.dims if dim not in from_dims]
        perm = [self.dims.index(dim) for dim in other_dims + from_dims]
        tensor = self.tensor.permute(perm)
        if len(from_dims) == 2:
            return Dense(tensor=tensor.diagonal(dim1=-2, dim2=-1), dims=other_dims + [to_dim])

        diag_index = torch.arange(to_dim.length, device=tensor.device)
        index = (...,) + tuple(diag_index for _ in from_dims)
        return Dense(tensor=tensor[index], dims=other_dims + [to_dim])

    def diagonal_embed(self, from_dim: Dim, to_dims: list[Dim]) -> "Dense":
        assert from_dim in self.dims, f"Cannot embed diagonal from dim {from_dim} that is not in {self.dims}."
        assert len(to_dims) > 0, "Must provide at least one dim to diagonal_embed."
        assert len(set(to_dims)) == len(to_dims), "Cannot diagonal_embed to repeated dims."
        assert all(dim not in self.dims for dim in to_dims), \
            f"Cannot embed diagonal to dims {to_dims} that already appear in {self.dims}."
        assert all(dim.length == from_dim.length for dim in to_dims), \
            f"All to_dims must have the same length as from_dim {from_dim}."

        if len(to_dims) == 1:
            dims = [to_dims[0] if dim is from_dim else dim for dim in self.dims]
            return Dense(tensor=self.tensor, dims=dims)

        other_dims = [dim for dim in self.dims if dim is not from_dim]
        perm = [self.dims.index(dim) for dim in other_dims + [from_dim]]
        tensor = self.tensor.permute(perm)
        if len(to_dims) == 2:
            return Dense(tensor=torch.diag_embed(tensor), dims=other_dims + to_dims)

        result_shape = [dim.length for dim in other_dims + to_dims]
        result = torch.zeros(result_shape, dtype=tensor.dtype, device=tensor.device)
        diag_index = torch.arange(from_dim.length, device=tensor.device)
        index = (...,) + tuple(diag_index for _ in to_dims)
        result[index] = tensor
        return Dense(tensor=result, dims=other_dims + to_dims)

    def replace_dims(self, dim_map: dict[Dim, Dim]) -> "Dense":
        new_dims = [dim_map.get(dim, dim) for dim in self.dims]
        return Dense(tensor=self.tensor, dims=new_dims)
    
    def apply(self, func: Callable[[torch.Tensor], torch.Tensor]) -> "Dense":
        return Dense(tensor=func(self.tensor), dims=self.dims)

    def norm(self, dims: list[Dim], p: Union[int, float] = 1) -> "Dense":
        product_of_dims = reduce(operator.mul, [d.length for d in dims if d not in self.dims], 1.0)
        if p != 1:
            product_of_dims = product_of_dims ** (1 / p)
        dims_idx = [self.dims.index(dim) for dim in dims if dim in self.dims]
        if len(dims_idx) > 0:
            tensor = torch.linalg.vector_norm(self.tensor, ord=p, dim=tuple(dims_idx))
        else:
            tensor = self.tensor.abs()
        remaining_dims = [dim for dim in self.dims if dim not in dims]
        return Dense(tensor=tensor * product_of_dims, dims=remaining_dims)

    
    def tensordot(self, other: "Dense", dims: list[Dim]) -> "Dense":
        # assert all(dim in self.dims and dim in other.dims for dim in dims), f"Cannot tensordot on dims {dims} that are not all in both tensors."
        result_dims = sorted(set(self.dims + other.dims) - set(dims))
        all_dims = sorted(set(self.dims + other.dims))
        dim_names = {dim: i for i, dim in enumerate(all_dims)}
        result_tensor = torch.einsum(
            self.tensor,
            [dim_names[dim] for dim in self.dims],
            other.tensor,
            [dim_names[dim] for dim in other.dims],
            [dim_names[dim] for dim in result_dims],
        )

        return Dense(tensor=result_tensor, dims=result_dims)
    
    def __str__(self):
        values = []
        values.append(f"mean={self.tensor.mean().item():.2g}")
        values.append(f"std={self.tensor.std().item():.2g}")
        values.append(f"min={self.tensor.min().item():.2g}")
        values.append(f"max={self.tensor.max().item():.2g}")
        for i in range(1 << len(self.dims)):
            tup = tuple(-1 if (i & (1 << j)) > 0 else 0 for j in range(len(self.dims)))
            a = "".join("1" if (i & (1 << j)) > 0 else "0" for j in range(len(self.dims)))
            values.append(f"{a}={self.tensor[tup].item():.2g}")
        values = ", ".join(values)

        return f"[{', '.join([f'{d}.{d.length}' for d in self.dims])}]({values})"

    def isfinite(self) -> bool:
        return Dense(torch.isfinite(self.tensor), dims=self.dims)
    
    def all(self) -> bool:
        return self.tensor.all().item()
    
@dataclass
class TN:
    factors: list[Dense]

    @property
    def shape(self) -> torch.Size:
        return torch.Size([dim.length for dim in self.dims])
    
    def _check_dtype(self):
        assert all(f.tensor.dtype == torch.float32 for f in self.factors), "Currently only float32 TNs are supported."

    def __post_init__(self):
        self.dtype = torch.float32 if len(self.factors) == 0 else self.factors[0].tensor.dtype
        assert all(f.tensor.dtype == self.dtype for f in self.factors), "All factors in a TN must have the same dtype."
        assert self.dtype == torch.float32, "Currently only float32 TNs are supported."
        # TODO: If a factor's dims are subset of another factor's dims, merge them. Using `expand` to expand the smaller factor to the larger factor's dims, and then multiply the tensors together. This will reduce the number of factors and make subsequent operations more efficient.`
        factors = list(self.factors)
        changed = True
        while changed:
            changed = False
            for i in range(len(factors)):
                dims_i = set(factors[i].dims)
                for j in range(len(factors)):
                    if i == j:
                        continue
                    dims_j = set(factors[j].dims)
                    if dims_i.issubset(dims_j):
                        merged = Dense(
                            tensor=factors[j].tensor * factors[i].expand(factors[j].dims),
                            dims=factors[j].dims,
                        )
                        factors[j] = merged
                        del factors[i]
                        changed = True
                        break
                if changed:
                    break
        self.factors = factors
        self.factors.sort(key=lambda f: f.dims)
        
        self.dims = list(sorted(set(dim for f in self.factors for dim in f.dims)))

    @staticmethod
    def from_dense(tensor: Union[Dense, torch.Tensor, int, float]) -> "TN":
        if not isinstance(tensor, Dense):
            tensor = Dense.new_from(tensor)
        return TN(factors=[tensor])
    
    def to_dense(self) -> Dense:
        # TODO: contract the tensor using torch.einsum
        if len(self.factors) == 0:
            return Dense(tensor=torch.tensor(1.0), dims=[])

        dim_names = {dim: i for i, dim in enumerate(self.dims)}
        args = []
        for factor in self.factors:
            args.extend([factor.tensor, [dim_names[dim] for dim in factor.dims]])
        tensor = torch.einsum(*args, [dim_names[dim] for dim in self.dims])
        return Dense(tensor=tensor, dims=list(self.dims))
    
    def _assert(self):
        assert self.dims == list(sorted(set(dim for f in self.factors for dim in f.dims))), "TN dims should be the sorted unique dims from all factors."

    def dim(self) -> int:
        return len(self.shape)
    
    def numel(self) -> int:
        return self.shape.numel()
    
    def real_numel(self) -> int:
        result = 0
        for f in self.factors:
            result += f.tensor.numel()
        return result
    
    def __str__(self):
        factors = []
        for f in self.factors:
            factors.append(f"{list(f.dims)}")
        return f"{' * '.join(factors)}"
    
    def clone(self) -> "TN":
        return TN(factors=[f.clone() for f in self.factors])

    def _scale(self, scalar: float):
        if len(self.factors) == 0:
            self.factors.append(Dense(tensor=torch.tensor(scalar), dims=[]))
        else:
            self.factors[0].tensor *= float(scalar)

    def __mul__(self, other: Union["TN", Dense, float, int]) -> "TN":
        if isinstance(other, Dense):
            other = TN.from_dense(other)
        if isinstance(other, (float, int)):
            res = self.clone()
            res._scale(other)
            return res
        else:
            return TN(factors=self.factors + other.factors)
    
    def __rmul__(self, other: Union[float, int]) -> "TN":
        return self.__mul__(other)
    
    def __add__(self, other: Union["TN", Dense, float, int]) -> "TN":
        if isinstance(other, Dense):
            other = TN.from_dense(other)
        dense = self.to_dense() + (other.to_dense() if isinstance(other, TN) else other)
        return TN.from_dense(dense)
    
    def __radd__(self, other: Union[Dense, float, int]) -> "TN":
        return self.__add__(other)
    
    def __sub__(self, other: Union["TN", Dense, float, int]) -> "TN":
        if isinstance(other, Dense):
            other = TN.from_dense(other)
        dense = self.to_dense() - (other.to_dense() if isinstance(other, TN) else other)
        return TN.from_dense(dense)
    
    def __rsub__(self, other: Union["TN", Dense, float, int]) -> "TN":
        if isinstance(other, Dense):
            other = TN.from_dense(other)
        dense = (other.to_dense() if isinstance(other, TN) else other) - self.to_dense()
        return TN.from_dense(dense)
    
    def __neg__(self) -> "TN":
        return self * -1
    
    def reciprocal(self) -> "TN":
        return TN(factors=[Dense(tensor=torch.reciprocal(f.tensor), dims=f.dims) for f in self.factors])
    
    def __truediv__(self, other: Union["TN", Dense, float, int]) -> "TN":
        if isinstance(other, Dense):
            other = TN.from_dense(other)
        if isinstance(other, (float, int)):
            return self * (1 / other)
        else:
            assert self.shape == other.shape, f"Cannot divide TNs with different shapes: {self.shape} vs {other.shape}."
            return self * other.reciprocal()
        
    def __rtruediv__(self, other: Union["TN", Dense, float, int]) -> "TN":
        if isinstance(other, Dense):
            other = TN.from_dense(other)
        if isinstance(other, (float, int)):
            return (1 / other) * self.reciprocal()
        else:
            assert self.shape == other.shape, f"Cannot divide TNs with different shapes: {self.shape} vs {other.shape}."
            return other * self.reciprocal()

    def sum(self, dims: list[Dim]) -> "TN":
        # TODO: contract the related tensor using torch.einsum
        # remove the contracted dims from the factors, and update the shape accordingly.
        unrelated_dims = set(dims) - set(self.dims)
        unrelated_dims_product = reduce(operator.mul, [d.length for d in unrelated_dims], 1.0)
        reduce_dims = set(dims) & set(self.dims)
        if len(reduce_dims) == 0:
            return self * unrelated_dims_product
        remaining_factors: list[Dense] = []
        related = []
        for factor in self.factors:
            if any(dim in reduce_dims for dim in factor.dims):
                related.append(factor)
            else:
                remaining_factors.append(factor.clone())

        related_dims = set(dim for factor in related for dim in factor.dims)
        output_dims = list(sorted(related_dims - reduce_dims))
        dim_names = {dim: i for i, dim in enumerate(sorted(related_dims))}
        args = []
        for factor in related:
            args.extend([factor.tensor, [dim_names[dim] for dim in factor.dims]])
        tensor = torch.einsum(*args, [dim_names[dim] for dim in output_dims])
        remaining_factors.append(Dense(tensor=tensor, dims=output_dims))

        return TN(factors=remaining_factors) * unrelated_dims_product
        
    
    def norm(self, dims: list[Dim], p: Union[int, float] = 1) -> "TN":
        remaining_factors: list[Dense] = []
        related: Dense = 1
        for factor in self.factors:
            if any(dim in dims for dim in factor.dims):
                related = factor * related
            else:
                remaining_factors.append(factor.apply(torch.abs))
        if isinstance(related, int) and related == 1:
            related = Dense(tensor=torch.tensor(1.0), dims=[])

        remaining_factors.append(related.norm(dims=dims, p=p))
        return TN(factors=remaining_factors)
        

    def apply_multiplicative(self, func: Callable[[torch.Tensor], torch.Tensor]) -> "TN":
        return TN(factors=[Dense(tensor=func(f.tensor), dims=f.dims) for f in self.factors])
    
    def index_reduce_sum(self, dim: Dim, index: Indices, target_dim: Dim) -> "TN":
        # TODO: materialize all factors that have `dim` into Dense, apply the index_reduce_sum to it, and return a new TN with that new factor.
        if dim not in self.dims:
            ones = torch.ones([dim.length])
            result = torch.zeros_like(ones)
            result.index_add_(0, index, ones)
            return self * Dense(tensor=result, dims=[target_dim])

        related = []
        remaining_factors = []
        for factor in self.factors:
            if dim in factor.dims:
                related.append(factor)
            else:
                remaining_factors.append(factor.clone())

        related_dims = list(sorted(set(dim for factor in related for dim in factor.dims)))
        dim_names = {dim: i for i, dim in enumerate(related_dims)}
        args = []
        for factor in related:
            args.extend([factor.tensor, [dim_names[dim] for dim in factor.dims]])
        tensor = torch.einsum(*args, [dim_names[dim] for dim in related_dims])
        dense = Dense(tensor=tensor, dims=related_dims)
        remaining_factors.append(dense.index_reduce_sum(dim, index, target_dim))
        return TN(factors=remaining_factors)
    
    def replace_dims(self, dim_map: dict[Dim, Dim]) -> "TN":
        return TN(factors=[f.replace_dims(dim_map) for f in self.factors])
    


__all__ = ["Dense", "TN", "Dim"]



        
