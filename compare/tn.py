
from dataclasses import dataclass
from typing import Any, Callable, Union

import torch

from boundlab.sparse.table import Indices

@dataclass
class Dense:
    tensor: torch.Tensor
    dims: list[Dim]

    def __post_init__(self):
        assert self.tensor.ndim == len(self.dims)
        assert len(set(self.dims)) == len(self.dims), "Dense tensors cannot repeat dims."
        for idx, dim in enumerate(self.dims):
            assert self.tensor.shape[idx] == dim.length, \
                f"Tensor axis {idx} has length {self.tensor.shape[idx]}, but dim has length {dim.length}."
        strides = self.tensor.stride()
        if 0 in strides:
            keep = [stride != 0 for stride in strides]
            self.tensor = self.tensor[tuple(slice(None) if keep_dim else 0 for keep_dim in keep)]
            self.dims = [dim for dim, keep_dim in zip(self.dims, keep) if keep_dim]
        assert all(self.tensor.stride(i) > 0 for i in range(self.tensor.ndim)), "Dense tensors must be contiguous in memory."
        # sort the dims and permute the tensor accordingly, so that the dims are always in sorted order.
        sorted_dims = list(sorted(self.dims))
        if self.dims != sorted_dims:
            perm = [self.dims.index(dim) for dim in sorted_dims]
            self.tensor = self.tensor.permute(perm)
            self.dims = sorted_dims

    @staticmethod
    def new_from(tensor: Union[torch.Tensor, "Dense", int, float]) -> "Dense":
        if isinstance(tensor, (int, float)):
            tensor = torch.tensor(tensor)
        if isinstance(tensor, Dense):
            tensor = tensor.tensor
        dims = [Dim(length=s, ordering=i) for i, s in enumerate(tensor.shape)]
        return Dense(tensor=tensor, dims=dims)

    def expand(self, dims: list[Dim]) -> torch.Tensor:
        assert set(self.dims).issubset(set(dims)), f"Cannot expand factor with dims {self.dims} to dims {dims}."
        # unsqueeze and expand and permute the tensor to match the new dims and shape.
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
        if dim not in self.dims:
            return self.clone()

        idx = self.dims.index(dim)
        shape = list(self.tensor.shape)
        shape[idx] = target_dim.length
        result = torch.zeros(shape, dtype=self.tensor.dtype, device=self.tensor.device)
        result.index_add_(idx, index, self.tensor)
        dims = list(self.dims)
        dims[idx] = target_dim
        return Dense(tensor=result, dims=dims)

    def align_from(self, tensor: Union[torch.Tensor, "Dense", int, float]) -> "Dense":
        if isinstance(tensor, (int, float)):
            tensor = torch.tensor(tensor)
        if isinstance(tensor, Dense):
            tensor = tensor.tensor
        assert tensor.ndim == len(self.dims)
        return Dense(tensor=tensor, dims=self.dims)
    
    def __add__(self, other: Union["Dense", int, float, torch.Tensor]) -> "Dense":
        if not isinstance(other, Dense):
            other = self.align_from(other)
        dims = list(sorted(set(self.dims) | set(other.dims)))
        return Dense(tensor=self.expand(dims) + other.expand(dims), dims=dims)
    
    def __mul__(self, other: Union["Dense", int, float, torch.Tensor]) -> "Dense":
        if not isinstance(other, Dense):
            other = self.align_from(other)
        dims = list(sorted(set(self.dims) | set(other.dims)))
        return Dense(tensor=self.expand(dims) * other.expand(dims), dims=dims)
    
    def __sub__(self, other: Union["Dense", int, float, torch.Tensor]) -> "Dense":
        if not isinstance(other, Dense):
            other = self.align_from(other)
        dims = list(sorted(set(self.dims) | set(other.dims)))
        return Dense(tensor=self.expand(dims) - other.expand(dims), dims=dims)
    
    def __rsub__(self, other: Union[int, float, torch.Tensor]) -> "Dense":
        if not isinstance(other, Dense):
            other = self.align_from(other)
        dims = list(sorted(set(self.dims) | set(other.dims)))
        return Dense(tensor=other.expand(dims) - self.expand(dims), dims=dims)
    
    def __truediv__(self, other: Union["Dense", int, float, torch.Tensor]) -> "Dense":
        if not isinstance(other, Dense):
            other = self.align_from(other)
        dims = list(sorted(set(self.dims) | set(other.dims)))
        return Dense(tensor=self.expand(dims) / other.expand(dims), dims=dims)

    
    def __neg__(self) -> "Dense":
        return Dense(tensor=-self.tensor, dims=self.dims)
    
    def sum(self, dims: list[Dim]) -> "Dense":
        tensor = self.tensor.sum(dim=[self.dims.index(dim) for dim in dims if dim in self.dims])
        dims = [dim for dim in self.dims if dim not in dims]
        return Dense(tensor=tensor, dims=dims)
    
    def allclose(self, other: "Dense", eps: float = 1e-5) -> bool:
        if set(self.dims) != set(other.dims):
            return False
        other_aligned = other.align_from(self.tensor)
        return torch.allclose(self.tensor, other_aligned.tensor, atol=eps)

@dataclass
class TN:
    factors: list[Dense]

    @property
    def shape(self) -> torch.Size:
        return torch.Size([dim.length for dim in self.dims])

    def __post_init__(self):
        # If a factor's dims are subset of another factor's dims, merge them. Using `expand` to expand the smaller factor to the larger factor's dims, and then multiply the tensors together. This will reduce the number of factors and make subsequent operations more efficient.`
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
        # contract the tensor using torch.einsum
        if len(self.factors) == 0:
            return Dense(tensor=torch.tensor(1.0), dims=[])

        dim_names = {dim: i for i, dim in enumerate(self.dims)}
        args = []
        for factor in self.factors:
            args.extend([factor.tensor, [dim_names[dim] for dim in factor.dims]])
        tensor = torch.einsum(*args, [dim_names[dim] for dim in self.dims])
        return Dense(tensor=tensor, dims=list(self.dims))
    
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
            factors.append(f"{list(f.dims)})")
        return f"<{' * '.join(factors)}>"
    
    def clone(self) -> "TN":
        return TN(factors=[f.clone() for f in self.factors])

    def _scale(self, scalar: float):
        if len(self.factors) == 0:
            self.factors.append(Dense(tensor=torch.tensor(scalar), dims=[]))
        else:
            self.factors[0].tensor *= scalar

    def _align_tn(self, other: Union["TN", torch.Tensor]) -> "TN":
        if isinstance(other, torch.Tensor):
            assert other.shape == self.shape, \
                f"Cannot align tensor with shape {other.shape} to TN with shape {self.shape}."
            return TN.from_dense(Dense(tensor=other, dims=list(self.dims)))

        assert self.shape == other.shape, \
            f"Cannot align TNs with different shapes: {self.shape} vs {other.shape}."
        if self.dims == other.dims:
            return other

        dim_map = {dim: self.dims[idx] for idx, dim in enumerate(other.dims)}
        return TN(factors=[
            Dense(tensor=f.tensor, dims=[dim_map[dim] for dim in f.dims])
            for f in other.factors
        ])

    def __mul__(self, other: Union["TN", float, int, torch.Tensor]) -> "TN":
        if isinstance(other, (float, int)):
            res = self.clone()
            res._scale(other)
            return res
        else:
            other = self._align_tn(other)
            assert self.shape == other.shape, f"Cannot multiply TNs with different shapes: {self.shape} vs {other.shape}."
            return TN(factors=self.factors + other.factors)
    
    def __rmul__(self, other: Union[float, int, torch.Tensor]) -> "TN":
        return self.__mul__(other)
    
    def __add__(self, other: Union["TN", float, int, torch.Tensor]) -> "TN":
        dense = self.to_dense() + (self._align_tn(other).to_dense() if isinstance(other, TN) else other)
        return TN.from_dense(dense)
    
    def __radd__(self, other: Union[float, int, torch.Tensor]) -> "TN":
        return self.__add__(other)
    
    def __sub__(self, other: Union["TN", float, int, torch.Tensor]) -> "TN":
        dense = self.to_dense() - (self._align_tn(other).to_dense() if isinstance(other, TN) else other)
        return TN.from_dense(dense)
    
    def __rsub__(self, other: Union[float, int, torch.Tensor]) -> "TN":
        dense = (self._align_tn(other).to_dense() if isinstance(other, TN) else other) - self.to_dense()
        return TN.from_dense(dense)
    
    def __neg__(self) -> "TN":
        return self * -1
    
    def reciprocal(self) -> "TN":
        return TN(factors=[Dense(tensor=torch.reciprocal(f.tensor), dims=f.dims) for f in self.factors])
    
    def __truediv__(self, other: Union["TN", float, int, torch.Tensor]) -> "TN":
        if isinstance(other, (float, int)):
            return self * (1 / other)
        else:
            other = self._align_tn(other)
            assert self.shape == other.shape, f"Cannot divide TNs with different shapes: {self.shape} vs {other.shape}."
            return self * other.reciprocal()
        
    def __rtruediv__(self, other: Union["TN", float, int, torch.Tensor]) -> "TN":
        if isinstance(other, (float, int)):
            return (1 / other) * self.reciprocal()
        else:
            other = self._align_tn(other)
            assert self.shape == other.shape, f"Cannot divide TNs with different shapes: {self.shape} vs {other.shape}."
            return other * self.reciprocal()

    def sum(self, dims: list[Dim]) -> "TN":
        # contract the related tensor using torch.einsum
        # remove the contracted dims from the factors, and update the shape accordingly.
        reduce_dims = set(dims) & set(self.dims)
        if len(reduce_dims) == 0:
            return self.clone()

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

        return TN(factors=remaining_factors)
        

    def apply_multiplicative(self, func: Callable[[torch.Tensor], torch.Tensor]) -> "TN":
        return TN(factors=[Dense(tensor=func(f.tensor), dims=f.dims) for f in self.factors])
    
    def index_reduce_sum(self, dim: Dim, index: Indices, target_dim: Dim) -> "TN":
        # TODO: materialize all factors that have `dim` into Dense, apply the index_reduce_sum to it, and return a new TN with that new factor.
        pass




        
