from dataclasses import dataclass
import copy
import warnings



from functools import reduce
import string

import torch

from boundlab import utils
from boundlab.linearop._base import LinearOp, ComposedOp, ScalarOp, SumOp
from boundlab.utils import merge_name, EQCondition
from boundlab.linearop._shape import ExpandOp, PermuteOp
from boundlab.utils import merge_name
import warnings


class EinsumOp(LinearOp):
    r"""A linear operator defined by an Einstein summation with a fixed tensor.

    Depending on ``input_dims`` and ``output_dims``, this can express
    contraction (dot-product-like behavior), Hadamard-style elementwise
    multiplication, and dimension expansion.
    """

    def __init__(self, tensor: torch.Tensor, input_dims: list[int], output_dims: list[int], name=None):
        """Initialize an EinsumOp.

        This operator behaves like contraction on ``input_dims - output_dims``,
        elementwise multiplication on ``input_dims & output_dims``, and
        expansion on ``output_dims - input_dims``.

        Args:
            tensor: The fixed tensor used in the Einstein summation.
            input_dims: A list of dimensions of `tensor` that correspond to the input tensor dimensions.
            output_dims: A list of dimensions of `tensor` that correspond to the output tensor dimensions.
            name: Optional name for display purposes.
        """
        from boundlab.linearop._base import LinearOpFlags
        self.tensor = tensor
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.dot_dims = [i for i in input_dims if i not in output_dims]
        self.mul_dims = [i for i in output_dims if i in input_dims]
        self.batch_dims = [i for i in output_dims if i not in input_dims]
        input_shape = torch.Size(tensor.shape[i] for i in input_dims)
        output_shape = torch.Size(tensor.shape[i] for i in output_dims)
        assert max(i for i in input_dims + output_dims) == tensor.dim() - 1, "input_dims and output_dims must be valid dimensions of the tensor"
        if name is not None:
            self.name = name

        super().__init__(input_shape, output_shape)
        if self.batch_dims == []:
            self.flags |= LinearOpFlags.IS_PURE_CONTRACTING
        if self.dot_dims == []:
            self.flags |= LinearOpFlags.IS_PURE_EXPANDING

    def __str__(self):
        if self.name:
            return self.name
        else:
            return f"<einsum {list(self.tensor.shape)}: {list(self.input_dims)} -> {list(self.output_dims)}>"
    # ---- einsum helpers ----

    def _einsum_strs(self):
        """Return (tensor_str, input_str, output_str) for einsum."""
        t_str = [i for i in range(self.tensor.dim())]
        i_str = [i for i in self.input_dims]
        o_str = [i for i in self.output_dims]
        return t_str, i_str, o_str

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t_idx = [i for i in range(self.tensor.dim())]
        return torch.einsum(self.tensor, t_idx, x, self.input_dims, self.output_dims)
    
    def backward(self, grad: torch.Tensor) -> torch.Tensor:
        t_idx = [i for i in range(self.tensor.dim())]
        return torch.einsum(self.tensor, t_idx, grad, self.output_dims, self.input_dims)
    
    def vforward(self, x: torch.Tensor) -> torch.Tensor:
        t_idx = [i for i in range(self.tensor.dim())]
        a_idx = [self.tensor.dim() + i for i in range(x.dim() - len(self.input_dims))]
        return torch.einsum(self.tensor, t_idx, x, self.input_dims + a_idx, self.output_dims + a_idx)
    
    def vbackward(self, grad: torch.Tensor) -> torch.Tensor:
        t_idx = [i for i in range(self.tensor.dim())]
        a_idx = [self.tensor.dim() + i for i in range(grad.dim() - len(self.output_dims))]
        return torch.einsum(self.tensor, t_idx, grad, a_idx + self.output_dims, a_idx + self.input_dims)

    def is_full(self) -> bool:
        """Check if this EinsumOp fully contracts all input dimensions (output_dims & input_dims == ø)."""
        return self.mul_dims == []

    def is_hardmard(self) -> bool:
        """Check whether this EinsumOp performs no contraction over input dims.

        Note:
            The method name keeps the historical ``hardmard`` spelling for
            backward compatibility.
        """
        return self.dot_dims == []

    def is_non_expanding(self) -> bool:
        """Check if this EinsumOp doesn't introduce new dimensions (output_dims - input_dims == ø)."""
        return self.batch_dims == []

    def is_tensordot(self) -> bool:
        """Check if this EinsumOp is effectively a tensordot (no elementwise multiplication)."""
        return all(self.tensor.stride(i) == 0 for i in self.mul_dims)
    
    def __neg__(self) -> "EinsumOp":
        """Return a new EinsumOp representing the negation of this operator."""
        return EinsumOp(-self.tensor, self.input_dims, self.output_dims, name=f"-{self}" if self.name and self.name else None)
        
    @staticmethod
    def from_hardmard(tensor: torch.Tensor, n_input_dims: int, name=None) -> "EinsumOp":
        """Create an EinsumOp for Hadamard-style multiplication with ``tensor``.

        Note:
            The constructor name keeps the historical ``hardmard`` spelling for
            backward compatibility.
        """
        output_dims = list(range(tensor.dim()))
        input_dims = output_dims[-n_input_dims:]
        return EinsumOp(tensor, input_dims, output_dims, name=name)
    
    @staticmethod
    def from_full(tensor: torch.Tensor, input_dim: int, name=None) -> "EinsumOp":
        """Create an EinsumOp that fully contracts over the specified input dimension."""
        output_dims = list(range(tensor.dim()-input_dim))
        input_dims = list(range(tensor.dim()-input_dim, tensor.dim()))
        return EinsumOp(tensor, input_dims, output_dims, name=name)

    def __mul__(self, scalar: float) -> "EinsumOp":
        """Scale the EinsumOp by a scalar."""
        return EinsumOp(self.tensor * scalar, self.input_dims, self.output_dims)

    def __rmul__(self, scalar: float) -> "EinsumOp":
        """Scale the EinsumOp by a scalar."""
        return self.__mul__(scalar)

    def squeeze_input(self, idx: int) -> "EinsumOp":
        """Drop ``input_dims[idx]`` whose corresponding tensor dim has size 1."""
        target = self.input_dims[idx]
        assert self.tensor.shape[target] == 1
        new_input_dims = self.input_dims[:idx] + self.input_dims[idx + 1:]
        if target in new_input_dims or target in self.output_dims:
            return EinsumOp(self.tensor, new_input_dims, self.output_dims, name=self.name)
        new_tensor = self.tensor.squeeze(target)
        adj = lambda d: d if d < target else d - 1
        return EinsumOp(new_tensor, [adj(d) for d in new_input_dims], [adj(d) for d in self.output_dims], name=self.name)

    def squeeze_output(self, idx: int) -> "EinsumOp":
        """Drop ``output_dims[idx]`` whose corresponding tensor dim has size 1."""
        target = self.output_dims[idx]
        assert self.tensor.shape[target] == 1
        new_output_dims = self.output_dims[:idx] + self.output_dims[idx + 1:]
        if target in new_output_dims or target in self.input_dims:
            return EinsumOp(self.tensor, self.input_dims, new_output_dims, name=self.name)
        new_tensor = self.tensor.squeeze(target)
        adj = lambda d: d if d < target else d - 1
        return EinsumOp(new_tensor, [adj(d) for d in self.input_dims], [adj(d) for d in new_output_dims], name=self.name)

    def unsqueeze_input(self, idx: int) -> "EinsumOp":
        """Insert a new size-1 input dim at position ``idx``."""
        new_dim = self.tensor.dim()
        new_tensor = self.tensor.unsqueeze(new_dim)
        new_input_dims = list(self.input_dims)
        new_input_dims.insert(idx, new_dim)
        return EinsumOp(new_tensor, new_input_dims, self.output_dims, name=self.name)

    def unsqueeze_output(self, idx: int) -> "EinsumOp":
        """Insert a new size-1 output dim at position ``idx``."""
        new_dim = self.tensor.dim()
        new_tensor = self.tensor.unsqueeze(new_dim)
        new_output_dims = list(self.output_dims)
        new_output_dims.insert(idx, new_dim)
        return EinsumOp(new_tensor, self.input_dims, new_output_dims, name=self.name)

    def _split_mul_dim(self, input_pos: int) -> "EinsumOp":
        """Split a mul_dim into a batch_dim (output-only) + a new size-1 input dim.

        Used when an expand broadcasts a size-1 input to size-k output (or
        vice-versa) at a shared dim.  After splitting, the original tensor dim
        stays in output_dims only, and a fresh size-1 tensor dim takes its
        place in input_dims.
        """
        old_t_dim = self.input_dims[input_pos]
        new_t_dim = self.tensor.dim()
        new_tensor = self.tensor.unsqueeze(new_t_dim)
        new_input_dims = list(self.input_dims)
        new_input_dims[input_pos] = new_t_dim
        return EinsumOp(new_tensor, new_input_dims, list(self.output_dims), name=self.name)

    def __matmul__(self, other: "LinearOp") -> "LinearOp":
        """Compose this EinsumOp with another LinearOp: (self ∘ other)(x) = self(other(x))."""
        if self.is_full():
            op = self.permute_for_input()
            idims = op.tensor.dim() - len(op.input_dims)
            assert all(a == b for a, b in zip(op.input_dims, range(idims, op.tensor.dim()))), "Full EinsumOp should have input_dims permuted to the end."
            tensor = other.vbackward(op.tensor)
            input_dims = list(range(idims, tensor.dim()))
            return EinsumOp(tensor, input_dims, op.output_dims, name=merge_name(self, "@", other))
        if isinstance(other, EinsumOp):
            return merge_einsumop(self, other)
        if isinstance(other, ScalarOp):
            if other.scalar == 1.0:
                return self
            return self * other.scalar
        return NotImplemented

    def __rmatmul__(self, other: "LinearOp") -> "LinearOp":
        """Compose another LinearOp with this EinsumOp: (other ∘ self)(x) = other(self(x))."""
        if self.is_full():
            op = self.permute_for_output()
            odims = len(op.output_dims)
            assert all(a == b for a, b in zip(op.output_dims, range(odims))), "Full EinsumOp should have output_dims permuted to the end."
            tensor = other.vforward(op.tensor)
            new_odims = len(other.output_shape)
            output_dims = list(range(0, new_odims))
            input_dims = list(range(new_odims, tensor.dim()))
            return EinsumOp(tensor, input_dims, output_dims, name=merge_name(other, "@", self))
        if isinstance(other, EinsumOp):
            return merge_einsumop(other, self)
        if isinstance(other, ScalarOp):
            return self * other.scalar
        return super().__rmatmul__(other)

    def __add__(self, other: "LinearOp") -> "LinearOp":
        """Add this EinsumOp to another LinearOp, returning a new LinearOp representing the sum."""
        if isinstance(other, LinearOp):
            return other.__radd__(self)
        return NotImplemented
    
    @staticmethod
    def from_scalar(scalarop: ScalarOp) -> "EinsumOp":
        """Convert a ScalarOp to an EinsumOp."""
        tensor = torch.tensor(scalarop.scalar).expand(scalarop.input_shape)
        return EinsumOp.from_hardmard(tensor, n_input_dims=tensor.dim(), name=scalarop.name)


    def __radd__(self, other: "LinearOp") -> "LinearOp":
        """Add another LinearOp to this EinsumOp."""
        if isinstance(other, ScalarOp):
            if other.scalar == 0.0:
                return self
            other = EinsumOp.from_scalar(other)
        if isinstance(other, EinsumOp):
            if self.input_shape == other.input_shape and self.output_shape == other.output_shape:
                self0 = self.permute_for_output()
                other0 = other.permute_for_output()
                if self0.input_dims == other0.input_dims and self0.output_dims == other0.output_dims:
                    return EinsumOp(self0.tensor + other0.tensor, self0.input_dims, self0.output_dims, name=f"{self} + {other0}")
                elif self.mul_conditions <= other.mul_conditions:
                    return self.purify_with(other)
                elif other.mul_conditions <= self.mul_conditions:
                    return self.purify_with(other)
            return super().__radd__(other)
        if isinstance(other, LinearOp):
            return super().__radd__(other)
        return NotImplemented

    def _indices_exec(self, indices: list[int], max_index: int) -> (list[int], list[int], int):
        result = [-1] * self.tensor.dim()
        for i, idx in zip(indices, self.input_dims):
            result[idx] = i
        for i in range(len(result)):
            if result[i] == -1:
                result[i] = max_index
                max_index += 1
        output_indices = [result[i] for i in self.output_dims]
        return result, output_indices, max_index

    def _indices_exec_reverse(self, indices: list[int], max_index: int) -> (list[int], list[int], int):
        result = [-1] * self.tensor.dim()
        for i, idx in zip(indices, self.output_dims):
            result[idx] = i
        for i in range(len(result)):
            if result[i] == -1:
                result[i] = max_index
                max_index += 1
        input_indices = [result[i] for i in self.input_dims]
        return result, input_indices, max_index

    def permute_for_input(self):
        permute_dims = [i for i in range(self.tensor.dim()) if i not in self.input_dims] + self.input_dims
        new_tensor = self.tensor.permute(permute_dims)
        input_dims = [permute_dims.index(i) for i in self.input_dims]
        output_dims = [permute_dims.index(i) for i in self.output_dims]
        len_diff = len(permute_dims) - len(self.input_dims)
        assert input_dims == list(range(len_diff, len(permute_dims))), "Input dimensions should be permuted to the end."
        return EinsumOp(new_tensor, input_dims, output_dims, name=self.name)

    def permute_for_output(self):
        non_output_dims = [i for i in range(self.tensor.dim()) if i not in self.output_dims]
        non_output_dims.sort(key=lambda x: self.input_dims.index(x))
        permute_dims = self.output_dims + non_output_dims
        new_tensor = self.tensor.permute(permute_dims)
        output_dims = [permute_dims.index(i) for i in self.output_dims]
        input_dims = []
        for i in self.input_dims:
            idx = i if i >= 0 else len(permute_dims) + i
            input_dims.append(permute_dims.index(idx))
        assert output_dims == list(range(len(output_dims))), "Output dimensions should be permuted to the end."
        return EinsumOp(new_tensor, input_dims, output_dims, name=self.name)
    
    def jacobian(self) -> torch.Tensor:
        """Return an explicit Jacobian for full-layout ``EinsumOp`` instances.

        Returns:
            A tensor with shape ``[*output_shape, *input_shape]`` when the
            operator is in full representation (``is_full()``). Otherwise
            returns ``NotImplemented``.
        """
        if self.is_full():
            return self.permute_for_output().tensor.view(self.output_shape + self.input_shape)
        else:
            # warnings.warn(f"Jacobian is not efficiently available for non-full EinsumOp {self}. Consider using `jacobian_scatter` or `force_jacobian` instead.", stacklevel=2)
            return self.force_jacobian()
            # raise NotImplementedError(f"Jacobian is only implemented for full EinsumOps: {self}")
        
    def abs(self) -> "LinearOp":
        """Return a LinearOp representing the element-wise absolute value of this EinsumOp."""
        return EinsumOp(self.tensor.abs(), self.input_dims, self.output_dims, name=f"|{self}|" if self.name else None)
    
    def sum_input(self):
        """Return a LinearOp that sums over the input dimensions of this EinsumOp."""
        tensor_dims_output_id = [None for i in range(self.tensor.dim())]
        for i, idx in enumerate(self.output_dims):
            tensor_dims_output_id[idx] = idx
        tensor_dims_output_id = [x for x in tensor_dims_output_id if x is not None]
        assert len(tensor_dims_output_id) == len(self.output_dims)
        assert len(self.output_dims) == len(self.tensor.shape) - len(self.dot_dims), "Output dims should cover all non-input dims of the tensor."
        
        if self.dot_dims == []:
            new_tensor = self.tensor
        else:
            new_tensor = self.tensor.sum(dim=self.dot_dims)
        assert len(new_tensor.shape) == len(self.output_dims), f" {self.tensor.shape} - {self.dot_dims} -> {new_tensor.shape}"
        new_input_dims = []
        new_output_dims = [tensor_dims_output_id.index(p) for p in self.output_dims]
        return EinsumOp(new_tensor, new_input_dims, new_output_dims, name=f"{self}.sum_input()" if self.name else None)
    
    def sum_output(self):
        """Return a LinearOp that sums over the output dimensions of this EinsumOp."""
        tensor_dims_input_id = [None for i in range(self.tensor.dim())]
        for i, idx in enumerate(self.input_dims):
            tensor_dims_input_id[idx] = idx
        tensor_dims_input_id = [x for x in tensor_dims_input_id if x is not None]
        
        new_tensor = self.tensor.sum(dim=self.batch_dims)
        new_input_dims = [tensor_dims_input_id.index(p) for p in self.input_dims]
        new_output_dims = []
        return EinsumOp(new_tensor, new_input_dims, new_output_dims, name=f"{self}.sum_output()" if self.name else None)

    def norm_input(self, p=1):
        """Return a LinearOp that computes the norm over the input dimensions of this EinsumOp."""
        tensor_dims_output_id = [None for i in range(self.tensor.dim())]
        for i, idx in enumerate(self.output_dims):
            tensor_dims_output_id[idx] = idx
        tensor_dims_output_id = [x for x in tensor_dims_output_id if x is not None]
        assert len(tensor_dims_output_id) == len(self.output_dims)
        assert len(self.output_dims) == len(self.tensor.shape) - len(self.dot_dims), "Output dims should cover all non-input dims of the tensor."
        
        if self.dot_dims == []:
            new_tensor = self.tensor.abs()
        else:
            new_tensor = self.tensor.norm(p, dim=self.dot_dims)
        assert len(new_tensor.shape) == len(self.output_dims), f" {self.tensor.shape} - {self.dot_dims} -> {new_tensor.shape}"
        new_input_dims = []
        new_output_dims = [tensor_dims_output_id.index(p) for p in self.output_dims]
        result = EinsumOp(new_tensor, new_input_dims, new_output_dims, name=f"{self}.norm_input(p={p})" if self.name else None)
        assert result.input_shape == torch.Size([]), f"Norm over input should have scalar input shape, got {result.input_shape}"
        assert result.output_shape == self.output_shape, f"Norm over input should preserve output shape, got {result.output_shape} vs {self.output_shape}"
        return result
    
    def norm_output(self, p=1):
        """Return a LinearOp that computes the norm over the output dimensions of this EinsumOp."""
        tensor_dims_input_id = [None for i in range(self.tensor.dim())]
        for i, idx in enumerate(self.input_dims):
            tensor_dims_input_id[idx] = idx
        tensor_dims_input_id = [x for x in tensor_dims_input_id if x is not None]
        
        new_tensor = self.tensor.norm(p, dim=self.batch_dims)
        new_input_dims = [tensor_dims_input_id.index(p) for p in self.input_dims]
        new_output_dims = []
        return EinsumOp(new_tensor, new_input_dims, new_output_dims, name=f"{self}.sum_output()" if self.name else None)

    @property
    def mul_conditions(self) -> "EQCondition":
        """Return a list of (input_dim, output_dim) pairs that are multiplied together in this EinsumOp."""
        sets = [[] for _ in range(self.tensor.dim())]
        for i, dim in enumerate(self.input_dims):
            sets[dim].append(i)
        for i, dim in enumerate(self.output_dims):
            sets[dim].append(~i)
        return EQCondition(set(tuple(s) for s in sets if len(s) > 1))
    
    def _get_dim(self, i) -> int:
        if i < 0:
            return self.output_dims[~i]
        else:
            return self.input_dims[i] 
    
    def add_conditions(self, target: EQCondition) -> "EinsumOp":
        """Return a new EinsumOp with additional (input_dim, output_dim) pairs multiplied together."""
        assert target >= self.mul_conditions, f"Cannot add {target} into {self.mul_conditions}"
        additional = target - self.mul_conditions
        
        result, maps = utils.multiple_diagnonal(self.tensor, [(self._get_dim(dim1), self._get_dim(dim2)) for dim1, dim2 in additional.to_pairs()])
        input_dims = [maps[self.input_dims[i]] for i in range(len(self.input_dims))]
        output_dims = [maps[self.output_dims[i]] for i in range(len(self.output_dims))]
        return EinsumOp(result, input_dims, output_dims, name=f"{self}.add_conditions({target})" if self.name else None)                                                 
    
    def remove_conditions(self, target: EQCondition) -> "EinsumOp":
        """Return a new EinsumOp with some (input_dim, output_dim) pairs no longer multiplied together."""
        assert target <= self.mul_conditions, f"Cannot remove {target} from {self.mul_conditions}"
        removed = self.mul_conditions - target
        assert removed.all_pairs()

        result, maps = utils.multiple_diag_embed(self.tensor, {self._get_dim(eqclass[0]): len(eqclass) for eqclass in removed.eqclasses})
        input_dims = [maps[self.input_dims[i]][0] for i in range(len(self.input_dims))]
        output_dims = [maps[self.output_dims[i]][0] for i in range(len(self.output_dims))]
        for eqclass in removed.eqclasses:
            for dim in eqclass[1:]:
                if dim < 0:
                    output_dims[~dim] = maps[self._get_dim(eqclass[0])].pop()
                else:
                    input_dims[dim] = maps[self._get_dim(eqclass[0])].pop()

        return EinsumOp(result, input_dims, output_dims, name=f"{self}.remove_conditions({target})" if self.name else None)

    def purify_with(self, other: "EinsumOp") -> "EinsumOp":
        if isinstance(other, ScalarOp):
            other = EinsumOp.from_scalar(other)
        if not isinstance(other, EinsumOp):
            if hasattr(other, "einsum_op"):
                other = other.einsum_op()
            else:
                raise NotImplementedError(f"purify_with is only implemented for EinsumOps: {self} vs {other}")
        assert self.input_shape == other.input_shape
        assert self.output_shape == other.output_shape
        
        if self.mul_conditions <= other.mul_conditions:
            self0 = self.permute_for_output()
            other0 = other.remove_conditions(self.mul_conditions).permute_for_output()
            assert self0.input_dims == other0.input_dims and self0.output_dims == other0.output_dims, f" {self0} {other0}"
            return self0 + other0
        elif other.mul_conditions <= self.mul_conditions:
            return self.remove_conditions(other.mul_conditions) + other
        else:
            intersection = self.mul_conditions + other.mul_conditions
            intersect_op = other.add_conditions(intersection)
            other_new = other - intersect_op.remove_conditions(other.mul_conditions)
            self_new = self + intersect_op.remove_conditions(self.mul_conditions)
            return self_new, other_new
            

def merge_einsumop(x: EinsumOp, y: EinsumOp) -> EinsumOp:
    assert x.input_shape == y.output_shape, f"Cannot compose EinsumOps with incompatible shapes: {x.output_shape} vs {y.input_shape}"
    output_idx = list(range(len(x.output_dims)))
    max_index = len(output_idx)

    x_idx, intermediate_idx, max_index = x._indices_exec_reverse(output_idx, max_index)
    assert len(x_idx) == x.tensor.dim()
    y_idx, input_idx, max_index = y._indices_exec_reverse(intermediate_idx, max_index)
    assert len(y_idx) == y.tensor.dim()

    new_tensor_idx = output_idx.copy()
    output_dims = list(range(len(output_idx)))
    input_dims = []
    for i in input_idx:
        if i in new_tensor_idx:
            input_dims.append(new_tensor_idx.index(i))
        else:
            new_tensor_idx.append(i)
            input_dims.append(len(new_tensor_idx) - 1)

    y_idx, x_idx, new_tensor_idx = _to_ascii_letters(y_idx, x_idx, new_tensor_idx)
    tensor = torch.einsum(f"{y_idx},{x_idx}->{new_tensor_idx}", y.tensor, x.tensor)
    return EinsumOp(tensor, input_dims, output_dims, name=merge_name(x, "@", y))

def _to_ascii_letters(*args: list[int]) -> tuple[str, ...]:
    return tuple("".join(string.ascii_letters[i] for i in a) for a in args)
    
