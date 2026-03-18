


import string

import torch

from boundlab.linearop._base import LinearOp, ComposedOp, SumOp


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
        self.dot_dims = list(set(input_dims) - set(output_dims))
        self.mul_dims = list(set(output_dims) & set(input_dims))
        self.batch_dims = list(set(output_dims) - set(input_dims))
        input_shape = torch.Size(tensor.shape[i] for i in input_dims)
        output_shape = torch.Size(tensor.shape[i] for i in output_dims)
        if name is not None:
            self.name = name
        else:
            self.name = f"<hdot {list(input_shape)} -> {list(output_shape)}>"
        super().__init__(input_shape, output_shape)

    # ---- einsum helpers ----

    def _einsum_strs(self):
        """Return (tensor_str, input_str, output_str) for einsum."""
        t_str = "".join(string.ascii_letters[i] for i in range(self.tensor.dim()))
        i_str = "".join(string.ascii_letters[i] for i in self.input_dims)
        o_str = "".join(string.ascii_letters[i] for i in self.output_dims)
        return t_str, i_str, o_str

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t_str, i_str, o_str = self._einsum_strs()
        return torch.einsum(f"{t_str},{i_str}->{o_str}", self.tensor, x)

    def backward(self, grad: torch.Tensor) -> torch.Tensor:
        t_str, i_str, o_str = self._einsum_strs()
        return torch.einsum(f"{t_str},{o_str}->{i_str}", self.tensor, grad)

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

    def is_zerotensor(self) -> bool:
        """Check if this EinsumOp is effectively a zero tensor."""
        return self.tensor.eq(0).all()

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
    def eye(shape: torch.Size) -> "EinsumOp":
        """Create an identity EinsumOp for the given shape."""
        tensor = torch.ones(shape)
        dims = list(range(len(shape)))
        return EinsumOp(tensor, dims, dims, name="I")

    def __mul__(self, scalar: float) -> "EinsumOp":
        """Scale the EinsumOp by a scalar."""
        return EinsumOp(self.tensor * scalar, self.input_dims, self.output_dims, name=f"{scalar} * {self}")

    def __rmul__(self, scalar: float) -> "EinsumOp":
        """Scale the EinsumOp by a scalar."""
        return self.__mul__(scalar)

    def __matmul__(self, other: "LinearOp") -> "LinearOp":
        """Compose this EinsumOp with another LinearOp: (self ∘ other)(x) = self(other(x))."""
        if isinstance(other, EinsumOp):
            return merge_einsumop(self, other)
        if isinstance(other, LinearOp):
            return ComposedOp(self, other)
        return NotImplemented

    def __rmatmul__(self, other: "LinearOp") -> "LinearOp":
        """Compose another LinearOp with this EinsumOp: (other ∘ self)(x) = other(self(x))."""
        if isinstance(other, EinsumOp):
            return merge_einsumop(other, self)
        if isinstance(other, LinearOp):
            return ComposedOp(other, self)
        return NotImplemented

    def __add__(self, other: "LinearOp") -> "LinearOp":
        """Add this EinsumOp to another LinearOp, returning a new LinearOp representing the sum."""
        if isinstance(other, EinsumOp):
            self0 = self.permute_for_output()
            other0 = other.permute_for_output()
            if self0.input_dims == other0.input_dims and self0.output_dims == other0.output_dims:
                return EinsumOp(self0.tensor + other0.tensor, self0.input_dims, self0.output_dims, name=f"{self} + {other0}")
        if isinstance(other, LinearOp):
            return SumOp(self, other)
        return NotImplemented

    def __radd__(self, other: "LinearOp") -> "LinearOp":
        """Add another LinearOp to this EinsumOp."""
        if isinstance(other, EinsumOp):
            return other.__add__(self)
        if isinstance(other, LinearOp):
            return SumOp(other, self)
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

def merge_einsumop(x: EinsumOp, y: EinsumOp) -> EinsumOp:
    output_idx = list(range(len(x.output_dims)))
    max_index = len(output_idx)

    x_idx, intermediate_idx, max_index = y._indices_exec_reverse(output_idx, max_index)
    y_idx, input_idx, max_index = x._indices_exec_reverse(intermediate_idx, max_index)

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
    return EinsumOp(tensor, input_dims, output_dims, name=f"({x} ⊚ {y})")

def _to_ascii_letters(*args: list[int]) -> tuple[str, ...]:
    return tuple("".join(string.ascii_letters[i] for i in a) for a in args)
