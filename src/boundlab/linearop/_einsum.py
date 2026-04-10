import warnings



from functools import reduce
import string

import torch

from boundlab.linearop._base import LinearOp, ComposedOp, ScalarOp, SumOp
from boundlab.linearop._shape import ExpandOp, PermuteOp
from boundlab.utils import merge_name


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
        self.tensor = tensor
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.dot_dims = [i for i in input_dims if i not in output_dims]
        self.mul_dims = [i for i in output_dims if i in input_dims]
        self.batch_dims = [i for i in output_dims if i not in input_dims]
        input_shape = torch.Size(tensor.shape[i] for i in input_dims)
        output_shape = torch.Size(tensor.shape[i] for i in output_dims)
        if name is not None:
            self.name = name
        else:
            self.name = None
        super().__init__(input_shape, output_shape)

    def __str__(self):
        if self.name is not None:
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
        if isinstance(other, PermuteOp):
            assert other.output_shape == self.input_shape, "PermuteOp shape must match EinsumOp output shape."
            new_input_dims = [self.input_dims[other.inv_dims[i]] for i in range(len(self.input_dims))]
            return EinsumOp(self.tensor, new_input_dims, self.output_dims, name=merge_name(self, "@", other))
        if isinstance(other, LinearOp):
            return NotImplemented
        return NotImplemented

    def __rmatmul__(self, other: "LinearOp") -> "LinearOp":
        """Compose another LinearOp with this EinsumOp: (other ∘ self)(x) = other(self(x))."""
        if self.is_full():
            op = self.permute_for_output()
            odims = len(op.output_dims)
            assert all(a == b for a, b in zip(op.output_dims, range(odims))), "Full EinsumOp should have output_dims permuted to the end."
            tensor = other.vforward(op.tensor)
            output_dims = list(range(0, odims))
            input_dims = list(range(odims, tensor.dim()))
            return EinsumOp(tensor, input_dims, output_dims, name=merge_name(other, "@", self))
        if isinstance(other, EinsumOp):
            return merge_einsumop(other, self)
        if isinstance(other, ScalarOp):
            return self * other.scalar
        if isinstance(other, PermuteOp):
            assert other.input_shape == self.output_shape, "PermuteOp shape must match EinsumOp output shape."
            new_output_dims = [self.output_dims[other.dims[i]] for i in range(len(self.output_dims))]
            return EinsumOp(self.tensor, self.input_dims, new_output_dims, name=merge_name(other, "@", self))
        from boundlab.linearop._shape import ExpandOp
        if isinstance(other, ExpandOp) and len(other.output_shape) == len(other.input_shape):
            assert other.input_shape == self.output_shape, "ExpandOp shape must match EinsumOp output shape."
            shape = list(self.tensor.shape)
            flag = True
            # Account for new leading dims by slicing other.output_shape
            for idx, size in zip(self.output_dims, other.output_shape):
                if shape[idx] != size:
                    assert shape[idx] == 1, f"Dimension {idx} has size {shape[idx]} but expected {size}"
                    shape[idx] = size
                    flag = flag and idx in self.batch_dims
            if flag:
                tensor = self.tensor.expand(shape)
                return EinsumOp(tensor, self.input_dims, self.output_dims, name=merge_name(other, "@", self))
        if isinstance(other, LinearOp):
            return super().__rmatmul__(other)
        return NotImplemented

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
            self0 = self.permute_for_output()
            other0 = other.permute_for_output()
            if self0.input_dims == other0.input_dims and self0.output_dims == other0.output_dims:
                return EinsumOp(self0.tensor + other0.tensor, self0.input_dims, self0.output_dims, name=f"{self} + {other0}")
            if other.is_full():
                out = self.jacobian_scatter(other.permute_for_output().tensor)
                return EinsumOp.from_full(out, len(self.input_dims), name=merge_name(self, "+", other))
            if self.is_full():
                out = other.jacobian_scatter(self.permute_for_output().tensor)
                return EinsumOp.from_full(out, len(self.input_dims), name=merge_name(self, "+", other))
            warnings.warn(f"Adding EinsumOps with different input/output dims: {self0} + {other0}. This needs `force_jacobian`.", stacklevel=2)
            return EinsumOp.from_full(SumOp(self, other).jacobian(), len(self.input_dims), name=merge_name(self, "+", other))
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
        permute_dims = self.output_dims + [i for i in range(self.tensor.dim()) if i not in self.output_dims]
        new_tensor = self.tensor.permute(permute_dims)
        output_dims = [permute_dims.index(i) for i in self.output_dims]
        input_dims = [permute_dims.index(i) for i in self.input_dims]
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
            warnings.warn(f"Jacobian is not efficiently available for non-full EinsumOp {self}. Consider using `jacobian_scatter` or `force_jacobian` instead.", stacklevel=2)
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
        return EinsumOp(new_tensor, new_input_dims, new_output_dims, name=f"{self}.sum_input()" if hasattr(self, "name") else None)
    
    def sum_output(self):
        """Return a LinearOp that sums over the output dimensions of this EinsumOp."""
        tensor_dims_input_id = [None for i in range(self.tensor.dim())]
        for i, idx in enumerate(self.input_dims):
            tensor_dims_input_id[idx] = idx
        tensor_dims_input_id = [x for x in tensor_dims_input_id if x is not None]
        
        new_tensor = self.tensor.sum(dim=self.batch_dims)
        new_input_dims = [tensor_dims_input_id.index(p) for p in self.input_dims]
        new_output_dims = []
        return EinsumOp(new_tensor, new_input_dims, new_output_dims, name=f"{self}.sum_output()" if hasattr(self, "name") else None)

    def norm_input(self, p=1):
        """Return a LinearOp that computes the norm over the input dimensions of this EinsumOp."""
        if p % 2 == 0:
            tensor0 = self.tensor.pow(p)
        else:
            if p == 1:
                tensor0 = self.tensor.abs()
            else:
                tensor0 = self.tensor.abs().pow(p)
        op = EinsumOp(tensor0, self.input_dims, self.output_dims).sum_input()
        return EinsumOp(op.tensor.pow(1/p) if p != 1 else op.tensor, op.input_dims, op.output_dims, name=f"{self}.norm_input(p={p})" if hasattr(self, "name") else None)
        
    def norm_output(self, p=1):
        """Return a LinearOp that computes the norm over the output dimensions of this EinsumOp."""
        if p % 2 == 0:
            tensor0 = self.tensor.pow(p)
        else:
            tensor0 = self.tensor.abs().pow(p)
        op = EinsumOp(tensor0, self.input_dims, self.output_dims).sum_output()
        return EinsumOp(op.tensor.pow(1/p), op.input_dims, op.output_dims, name=f"{self}.norm_output(p={p})" if hasattr(self, "name") else None)

    def jacobian_scatter(self, src: torch.Tensor) -> torch.Tensor:
        """Accumulate this operator's Jacobian into ``src`` efficiently.

        Args:
            src: Accumulator tensor with shape
                ``[*output_shape, *input_shape]``.

        Returns:
            A tensor with the same shape as ``src`` equal to
            ``src + jacobian(self)``.

        Notes:
            For full-layout operators, this reduces to direct addition.
            For compressed/masked layouts, this method performs a structured
            diagonal-scatter update that avoids full Jacobian expansion in the
            common case.
        """
        if self.is_full():
            return src + self.jacobian()
        else:
            mul_dims_numel = reduce(lambda x, y: x * y, (self.tensor.shape[i] for i in self.mul_dims), 1)
            batch_sizes = [self.tensor.shape[i] for i in self.batch_dims]
            dot_sizes = [self.tensor.shape[i] for i in self.dot_dims]

            bmd_dims = self.batch_dims + self.mul_dims + self.dot_dims
            bmd_inputs = [bmd_dims.index(i) for i in self.input_dims]
            bmd_outputs = [bmd_dims.index(i) for i in self.output_dims]

            output_permute = [None for i in range(len(bmd_outputs))]
            for i, idx in enumerate(bmd_outputs):
                output_permute[idx] = i
            input_permute = [None for i in range(len(bmd_inputs))]
            for i, idx in enumerate(bmd_inputs):
                input_permute[idx - len(self.batch_dims)] = i

            src_permute = output_permute + [i + len(bmd_outputs) for i in input_permute] # bmmd

            permuted = src.permute(src_permute)
            permuted_shape = permuted.shape
            permuted = permuted.view(*batch_sizes, mul_dims_numel, mul_dims_numel, *dot_sizes)

            bdm_dims = self.batch_dims + self.dot_dims + self.mul_dims
            tensor = self.tensor.permute(bdm_dims).view(*batch_sizes, *dot_sizes, mul_dims_numel)
            existing_diag = torch.diagonal(permuted, dim1=len(self.batch_dims), dim2=len(self.batch_dims) + 1)
            scattered = torch.diagonal_scatter(permuted, existing_diag + tensor, dim1=len(self.batch_dims), dim2=len(self.batch_dims) + 1)
            scattered = scattered.view(permuted_shape)

            src_permute_inv = [None for i in range(len(scattered.shape))]
            for i, _ in enumerate(scattered.shape):
                src_permute_inv[src_permute[i]] = i

            result = scattered.permute(src_permute_inv)
            assert result.shape == src.shape, f"Expected scattered shape {src.shape}, got {result.shape}."
            assert torch.allclose(result, src + self.force_jacobian(), atol=1e-5), f"Scattered result does not match expected Jacobian scatter: {result} vs {src + self.jacobian()}"
            return result

            

def merge_einsumop(x: EinsumOp, y: EinsumOp) -> EinsumOp:
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
    