"""Shape and indexing LinearOp implementations for bound propagation.

Each class implements explicit forward (the shape operation) and backward
(its adjoint/transpose) so that no automatic VJP is needed.
"""

from functools import reduce

import torch

from boundlab.linearop._base import LinearOp, LinearOpFlags, ScalarOp
from boundlab.utils import merge_name


def _meta_output_shape(fn, input_shape: torch.Size) -> torch.Size:
    """Compute output shape by tracing *fn* on a meta-device tensor."""
    return fn(torch.empty(input_shape, device="meta")).shape


# ---------------------------------------------------------------------------
# Reshape / view operations
# ---------------------------------------------------------------------------

class ReshapeOp(LinearOp):
    """Reshape (view) the input tensor to *target_shape*."""

    def __init__(self, input_shape: torch.Size, target_shape: tuple[int, ...]):
        self.target_shape = target_shape
        output_shape = _meta_output_shape(lambda x: x.reshape(*target_shape), input_shape)
        self.reshape_groups = []
        self.dims_map = {}
        self.dims_map_inv = {}
        i, j = 0, 0
        input_win = 0
        output_win = 0

        while True:
            input_numel = reduce(lambda x, y: x * y, input_shape[input_win:i+1], 1)
            output_numel = reduce(lambda x, y: x * y, output_shape[output_win:j+1], 1)
            if input_numel == output_numel:
                if i > input_win or j > output_win:
                    self.reshape_groups.append((input_win, i, output_win, j))
                else:
                    assert i == input_win and j == output_win
                    self.dims_map[i] = j
                    self.dims_map_inv[j] = i
                i += 1
                j += 1
                input_win = i
                output_win = j

            if input_numel < output_numel:
                i += 1
            
            if input_numel > output_numel:
                j += 1

            if i >= len(input_shape) or j >= len(output_shape):
                # Consume any remaining size-1 dims on either side
                while i < len(input_shape) and input_shape[i] == 1:
                    i += 1
                while j < len(output_shape) and output_shape[j] == 1:
                    j += 1
                if input_win < i or output_win < j:
                    self.reshape_groups.append((input_win, max(i - 1, input_win), output_win, max(j - 1, output_win)))
                assert i == len(input_shape) and j == len(output_shape), \
                    f"ReshapeOp: cannot align {input_shape} -> {output_shape} (i={i}, j={j})"
                break

        super().__init__(input_shape, output_shape, flags=LinearOpFlags.IS_NON_NEGATIVE | LinearOpFlags.IS_PURE_EXPANDING | LinearOpFlags.IS_PURE_CONTRACTING)

    def forward(self, x):
        return x.reshape(*self.target_shape)

    def backward(self, grad):
        return grad.reshape(self.input_shape)

    def vforward(self, x):
        extra = x.shape[len(self.input_shape):]
        return x.reshape(*self.output_shape, *extra)

    def vbackward(self, grad):
        extra = grad.shape[:-len(self.output_shape)]
        return grad.reshape(*extra, *self.input_shape)

    def __matmul__(self, other):
        from ._einsum import EinsumOp
        if isinstance(other, ReshapeOp):
            # Fuse consecutive reshapes into one reshape from self.input_shape to other.output_shape
            return ReshapeOp(self.input_shape, other.output_shape)
        if isinstance(other, EinsumOp):
            for in_s, in_e, out_s, out_e in self.reshape_groups:
                if any(i in other.mul_dims for i in other.output_dims[in_s:in_e + 1]):
                    return NotImplemented
            op = other.permute_for_output()
            assert all(i == p for i, p in enumerate(op.output_dims))

            tensor = self.vforward(op.tensor)
            def dims_map(d):
                shift = len(self.output_shape) - len(self.input_shape)
                return self.dims_map[d] if d < len(self.input_shape) else d + shift
            
            output_dims = list(range(len(self.output_shape)))
            input_dims = [dims_map(d) for d in op.input_dims]
            return EinsumOp(tensor, input_dims, output_dims, name=merge_name(self, "@", other))
        return NotImplemented

    def __rmatmul__(self, other):
        from ._einsum import EinsumOp
        if isinstance(other, EinsumOp):
            for in_s, in_e, out_s, out_e in self.reshape_groups:
                if any(i in other.mul_dims for i in other.input_dims[out_s:out_e + 1]):
                    return super().__rmatmul__(other)
            op = other.permute_for_input()
            n_non_input = op.tensor.dim() - len(op.input_dims)
            assert all(i + n_non_input == p for i, p in enumerate(op.input_dims))

            tensor = self.vbackward(op.tensor)
            def dims_map_inv(d):
                return self.dims_map_inv[d] if d >= 0 else d
            
            input_dims = list(range(n_non_input, n_non_input + len(self.input_shape)))
            output_dims = [dims_map_inv(d - n_non_input) + n_non_input for d in op.output_dims]
            return EinsumOp(tensor, input_dims, output_dims, name=merge_name(other, "@", self))
        return super().__rmatmul__(other)

    def __str__(self):
        return f"<reshape {list(self.input_shape)} -> {list(self.target_shape)}>"


class FlattenOp(LinearOp):
    """Flatten dimensions [start_dim .. end_dim] into a single dimension."""

    def __init__(self, input_shape: torch.Size, start_dim: int = 0, end_dim: int = -1):
        self.start_dim = start_dim
        self.end_dim = end_dim if end_dim >= 0 else len(input_shape) + end_dim
        self.original_sizes = input_shape[self.start_dim:self.end_dim + 1]
        output_shape = _meta_output_shape(
            lambda x: x.flatten(start_dim, end_dim), input_shape)
        super().__init__(input_shape, output_shape, flags=LinearOpFlags.IS_NON_NEGATIVE | LinearOpFlags.IS_PURE_EXPANDING | LinearOpFlags.IS_PURE_CONTRACTING)

    def forward(self, x):
        return x.flatten(self.start_dim, self.end_dim)

    def backward(self, grad):
        return grad.unflatten(self.start_dim, self.original_sizes)

    def __str__(self):
        return f"<flatten {self.start_dim} {self.end_dim}>"


class UnflattenOp(LinearOp):
    """Unflatten dimension *dim* into *sizes*."""

    def __init__(self, input_shape: torch.Size, dim: int, sizes: tuple[int, ...]):
        self.dim = dim
        self.sizes = sizes
        self.end_dim = dim + len(sizes) - 1
        output_shape = _meta_output_shape(
            lambda x: x.unflatten(dim, sizes), input_shape)
        super().__init__(input_shape, output_shape, flags=LinearOpFlags.IS_NON_NEGATIVE | LinearOpFlags.IS_PURE_EXPANDING | LinearOpFlags.IS_PURE_CONTRACTING)

    def forward(self, x):
        return x.unflatten(self.dim, self.sizes)

    def backward(self, grad):
        return grad.flatten(self.dim, self.end_dim)

    def __str__(self):
        return f"<unflatten {self.dim} {list(self.sizes)}>"


# ---------------------------------------------------------------------------
# Permutation / transposition
# ---------------------------------------------------------------------------

class PermuteOp(LinearOp):
    """Permute dimensions of the input tensor."""

    def __init__(self, input_shape: torch.Size, dims: tuple[int, ...]):
        self.dims = list(dims)
        self.inv_dims = [0] * len(dims)
        for i, d in enumerate(dims):
            self.inv_dims[d] = i
        output_shape = torch.Size(input_shape[d] for d in dims)
        super().__init__(input_shape, output_shape, flags=LinearOpFlags.IS_NON_NEGATIVE | LinearOpFlags.IS_PURE_EXPANDING | LinearOpFlags.IS_PURE_CONTRACTING)

    def forward(self, x):
        return x.permute(*self.dims)

    def backward(self, grad):
        return grad.permute(*self.inv_dims)

    def vforward(self, x):
        n = len(self.dims)
        batch_ndim = x.dim() - n
        perm = self.dims + [n + i for i in range(batch_ndim)]
        return x.permute(*perm)

    def vbackward(self, grad):
        n = len(self.inv_dims)
        batch_ndim = grad.dim() - n
        perm = list(range(batch_ndim)) + [batch_ndim + d for d in self.inv_dims]
        return grad.permute(*perm)

    def __matmul__(self, other):
        from ._einsum import EinsumOp
        if isinstance(other, EinsumOp):
            assert self.input_shape == other.output_shape
            new_output_dims = [other.output_dims[self.dims[i]] for i in range(len(other.output_dims))]
            return EinsumOp(other.tensor, other.input_dims, new_output_dims, name=merge_name(self, "@", other))
        return NotImplemented

    def __rmatmul__(self, other):
        from ._einsum import EinsumOp
        if isinstance(other, EinsumOp):
            assert self.output_shape == other.input_shape
            new_input_dims = [other.input_dims[self.inv_dims[i]] for i in range(len(other.input_dims))]
            return EinsumOp(other.tensor, new_input_dims, other.output_dims, name=merge_name(other, "@", self))
        return super().__rmatmul__(other)

    def __str__(self):
        return f"<permute {self.dims}>"

class TransposeOp(PermuteOp):
    """Swap two dimensions of the input tensor — special case of PermuteOp."""

    def __init__(self, input_shape: torch.Size, dim0: int, dim1: int):
        self.dim0 = dim0
        self.dim1 = dim1
        dims = list(range(len(input_shape)))
        dims[dim0], dims[dim1] = dims[dim1], dims[dim0]
        super().__init__(input_shape, tuple(dims))

    def __str__(self):
        return f"<transpose {self.dim0} {self.dim1}>"


# ---------------------------------------------------------------------------
# Squeeze / unsqueeze
# ---------------------------------------------------------------------------

class SqueezeOp(LinearOp):
    """Remove size-1 dimension(s)."""

    def __init__(self, input_shape: torch.Size, dim=None):
        self.dim = dim
        if dim is not None:
            self._is_noop = (input_shape[dim] != 1)
            if self._is_noop:
                output_shape = input_shape
            else:
                output_shape = torch.Size(
                    s for i, s in enumerate(input_shape) if i != dim)
        else:
            self._is_noop = all(s != 1 for s in input_shape)
            self._squeezed_dims = [i for i, s in enumerate(input_shape) if s == 1]
            output_shape = torch.Size(s for s in input_shape if s != 1)
        super().__init__(input_shape, output_shape, flags=LinearOpFlags.IS_NON_NEGATIVE | LinearOpFlags.IS_PURE_EXPANDING | LinearOpFlags.IS_PURE_CONTRACTING)

    def forward(self, x):
        return x.squeeze(self.dim) if self.dim is not None else x.squeeze()

    def backward(self, grad):
        if self._is_noop:
            return grad
        if self.dim is not None:
            return grad.unsqueeze(self.dim)
        for d in self._squeezed_dims:
            grad = grad.unsqueeze(d)
        return grad

    def __matmul__(self, other):
        """Fuse squeeze @ einsum: drop size-1 output dims."""
        from ._einsum import EinsumOp
        if isinstance(other, EinsumOp):
            assert self.input_shape == other.output_shape
            if self.dim is not None:
                squeezed = [] if self._is_noop else [self.dim]
            else:
                squeezed = list(self._squeezed_dims)
            op = other
            for pos in sorted(squeezed, reverse=True):
                op = op.squeeze_output(pos)
            return EinsumOp(op.tensor, op.input_dims, op.output_dims, name=merge_name(self, "@", other))
        return NotImplemented

    def __rmatmul__(self, other):
        """Fuse einsum @ squeeze: re-insert size-1 input dims."""
        from ._einsum import EinsumOp
        if isinstance(other, EinsumOp):
            assert self.output_shape == other.input_shape
            if self.dim is not None:
                squeezed = [] if self._is_noop else [self.dim]
            else:
                squeezed = list(self._squeezed_dims)
            op = other
            for pos in sorted(squeezed):
                op = op.unsqueeze_input(pos)
            return EinsumOp(op.tensor, op.input_dims, op.output_dims, name=merge_name(other, "@", self))
        return super().__rmatmul__(other)

    def __str__(self):
        return f"<squeeze {self.dim}>"


class UnsqueezeOp(LinearOp):
    """Insert a size-1 dimension at *dim*."""

    def __init__(self, input_shape: torch.Size, dim: int):
        if dim < 0:
            dim += len(input_shape) + 1
        assert 0 <= dim <= len(input_shape), f"Invalid unsqueeze dim {dim} for input shape {input_shape}"
        self.dim = dim
        output_shape = _meta_output_shape(lambda x: x.unsqueeze(dim), input_shape)
        super().__init__(input_shape, output_shape, flags=LinearOpFlags.IS_NON_NEGATIVE | LinearOpFlags.IS_PURE_EXPANDING | LinearOpFlags.IS_PURE_CONTRACTING)

    def forward(self, x):
        return x.unsqueeze(self.dim)

    def backward(self, grad):
        return grad.squeeze(self.dim)

    def vforward(self, x):
        return x.unsqueeze(self.dim)

    def vbackward(self, grad):
        batch_ndim = grad.dim() - len(self.output_shape)
        return grad.squeeze(batch_ndim + self.dim)

    def __matmul__(self, other):
        """Fuse unsqueeze @ einsum: insert a new size-1 output dim."""
        from ._einsum import EinsumOp
        if isinstance(other, EinsumOp):
            assert self.input_shape == other.output_shape, f"{self}, {other}"
            op = other.unsqueeze_output(self.dim)
            return EinsumOp(op.tensor, op.input_dims, op.output_dims, name=merge_name(self, "@", other))
        return NotImplemented

    def __rmatmul__(self, other):
        """Fuse einsum @ unsqueeze: drop the size-1 input dim."""
        from ._einsum import EinsumOp
        if isinstance(other, EinsumOp):
            assert self.output_shape == other.input_shape
            op = other.squeeze_input(self.dim)
            return EinsumOp(op.tensor, op.input_dims, op.output_dims, name=merge_name(other, "@", self))
        return super().__rmatmul__(other)

    def __str__(self):
        return f"<unsqueeze {list(self.input_shape)} {self.dim}>"


# ---------------------------------------------------------------------------
# Expand / repeat / tile
# ---------------------------------------------------------------------------

class ExpandOp(LinearOp):
    """Broadcast-expand dimensions (adjoint sums over expanded dims)."""

    def __new__(cls, input_shape: torch.Size, sizes: tuple[int, ...]):
        if input_shape == torch.Size(sizes):
            return ScalarOp(1.0, input_shape)
        
        return super().__new__(cls)

    def __init__(self, input_shape: torch.Size, sizes: tuple[int, ...]):
        self.sizes = sizes
        output_shape = _meta_output_shape(lambda x: x.expand(*sizes), input_shape)
        # Dims that need to be summed in backward
        n_new = len(output_shape) - len(input_shape)
        self.expand_indices: list[int] = list(range(n_new))
        for i in range(len(input_shape)):
            if input_shape[i] == 1 and output_shape[n_new + i] > 1:
                self.expand_indices.append(n_new + i)
        super().__init__(input_shape, output_shape, flags=LinearOpFlags.IS_NON_NEGATIVE | LinearOpFlags.IS_PURE_EXPANDING)

    def forward(self, x):
        return x.expand(*self.sizes)

    def backward(self, grad: torch.Tensor):
        if self.expand_indices:
            grad = grad.sum(dim=self.expand_indices, keepdim=True)

        n_new = len(self.output_shape) - len(self.input_shape)
        for i in range(n_new):
            grad = grad.squeeze(0)
        assert grad.shape == self.input_shape, f"{self} got {grad.shape}"
        return grad

    def __matmul__(self, other):
        """Fuse expand @ einsum: absorb broadcast output expansion into the tensor."""
        from ._einsum import EinsumOp
        if isinstance(other, EinsumOp):
            assert self.input_shape == other.output_shape
            n_new = len(self.output_shape) - len(self.input_shape)
            op = other
            # 1. Insert new leading output dims.
            for i in range(n_new):
                size = self.output_shape[i]
                new_dim = op.tensor.dim()
                new_tensor = op.tensor.unsqueeze(new_dim)
                if size > 1:
                    shape = list(new_tensor.shape)
                    shape[new_dim] = size
                    new_tensor = new_tensor.expand(shape)
                new_output_dims = [new_dim] + list(op.output_dims)
                op = EinsumOp(new_tensor, op.input_dims, new_output_dims, name=op.name)
            # 2. Expand existing dims that need size change.
            shape = list(op.tensor.shape)
            new_input_dims = list(op.input_dims)
            for i, t_dim in enumerate(op.output_dims):
                desired = self.output_shape[i]
                if shape[t_dim] == desired:
                    continue
                assert shape[t_dim] == 1, f"Dimension {t_dim} has size {shape[t_dim]} but expected 1"
                if t_dim in new_input_dims:
                    # Mul dim: split so this tensor dim becomes output-only.
                    inp_pos = new_input_dims.index(t_dim)
                    new_t_dim = len(shape)
                    shape.append(1)
                    new_input_dims[inp_pos] = new_t_dim
                shape[t_dim] = desired
            # Materialise any new tensor dims from splits.
            new_tensor = op.tensor
            n_extra = len(shape) - new_tensor.dim()
            for _ in range(n_extra):
                new_tensor = new_tensor.unsqueeze(-1)
            new_tensor = new_tensor.expand(shape)
            return EinsumOp(new_tensor, new_input_dims, op.output_dims, name=merge_name(self, "@", other))
        return NotImplemented

    def __rmatmul__(self, other):
        """Fuse einsum @ expand: absorb expand-broadcasting into the tensor."""
        from ._einsum import EinsumOp
        if isinstance(other, EinsumOp):
            assert self.output_shape == other.input_shape
            A = self.input_shape
            Ap = self.output_shape
            n_new = len(Ap) - len(A)
            to_handle: list[int] = []
            for p in range(len(Ap)):
                if p < n_new:
                    to_handle.append(p)
                else:
                    a = p - n_new
                    if A[a] == 1 and Ap[p] > 1:
                        to_handle.append(p)
            to_drop: list[int] = []   # new leading dims
            to_keep: list[int] = []   # broadcast dims (size 1 in A)
            for p in to_handle:
                if p < n_new:
                    to_drop.append(p)
                else:
                    to_keep.append(p)
            op = other
            # 1. Handle broadcast dims (keep position, just split mul if needed).
            for p in sorted(to_keep, reverse=True):
                t_dim = op.input_dims[p]
                if t_dim in op.output_dims:
                    op = op._split_mul_dim(p)
            # 2. Handle new leading dims (sum over tensor dim, remove position).
            for p in sorted(to_drop, reverse=True):
                t_dim = op.input_dims[p]
                if t_dim in op.output_dims:
                    op = op._split_mul_dim(p)
                    t_dim = op.input_dims[p]
                new_tensor = op.tensor.sum(dim=t_dim)
                new_input_dims = op.input_dims[:p] + op.input_dims[p + 1:]
                adj = lambda d, r=t_dim: d if d < r else d - 1
                new_input_dims = [adj(d) for d in new_input_dims]
                new_output_dims = [adj(d) for d in op.output_dims]
                op = EinsumOp(new_tensor, new_input_dims, new_output_dims, name=op.name)
            return EinsumOp(op.tensor, op.input_dims, op.output_dims, name=merge_name(other, "@", self))
        return super().__rmatmul__(other)

    def __str__(self):
        return f"<expand {list(self.input_shape)} -> {list(self.sizes)}>"


class RepeatOp(LinearOp):
    """Tile-repeat the tensor (adjoint folds and sums repeated blocks)."""

    def __init__(self, input_shape: torch.Size, sizes: tuple[int, ...]):
        self.sizes = sizes
        n_pad = len(sizes) - len(input_shape)
        self._padded_input_shape = torch.Size([1] * n_pad + list(input_shape))
        output_shape = torch.Size(
            s * r for s, r in zip(self._padded_input_shape, sizes))
        super().__init__(input_shape, output_shape, flags=LinearOpFlags.IS_NON_NEGATIVE | LinearOpFlags.IS_PURE_EXPANDING)

    def forward(self, x):
        return x.repeat(*self.sizes)

    def backward(self, grad):
        # Interleave (repeat_factor, original_size) pairs, then sum repeat dims
        new_shape = []
        for r, s in zip(self.sizes, self._padded_input_shape):
            new_shape.extend([r, s])
        grad = grad.reshape(new_shape)
        sum_dims = list(range(0, len(new_shape), 2))
        grad = grad.sum(dim=sum_dims)
        return grad.reshape(self.input_shape)

    def __str__(self):
        return f"<repeat {list(self.sizes)}>"


class TileOp(RepeatOp):
    """Alias for repeat with dimension-padding handled like ``torch.tile``."""

    def __init__(self, input_shape: torch.Size, sizes: tuple[int, ...]):
        # tile pads *sizes* with leading 1s when tensor has more dims
        n_pad = len(input_shape) - len(sizes)
        if n_pad > 0:
            sizes = (1,) * n_pad + tuple(sizes)
        super().__init__(input_shape, sizes)

    def __str__(self):
        return f"<tile {list(self.sizes)}>"


# ---------------------------------------------------------------------------
# Element-reordering (self-adjoint or simple inverse)
# ---------------------------------------------------------------------------

class FlipOp(LinearOp):
    """Reverse elements along *dims* (self-adjoint)."""

    def __init__(self, input_shape: torch.Size, dims):
        self.dims = dims
        super().__init__(input_shape, input_shape, flags=LinearOpFlags.IS_NON_NEGATIVE)

    def forward(self, x):
        return x.flip(self.dims)

    def backward(self, grad):
        return grad.flip(self.dims)

    def __str__(self):
        return f"<flip {self.dims}>"


class RollOp(LinearOp):
    """Circular-shift elements (adjoint is the inverse shift)."""

    def __init__(self, input_shape: torch.Size, shifts, dims):
        self.shifts = shifts
        self.dims = dims
        # Inverse shifts for backward
        if isinstance(shifts, int):
            self._inv_shifts = -shifts
        else:
            self._inv_shifts = [-s for s in shifts]
        super().__init__(input_shape, input_shape, flags=LinearOpFlags.IS_NON_NEGATIVE)

    def forward(self, x):
        return x.roll(self.shifts, self.dims)

    def backward(self, grad):
        return grad.roll(self._inv_shifts, self.dims)

    def __str__(self):
        return f"<roll {self.shifts} {self.dims}>"


# ---------------------------------------------------------------------------
# Diagonal
# ---------------------------------------------------------------------------

class DiagOp(LinearOp):
    """Extract or create a diagonal (1D↔2D)."""

    def __init__(self, input_shape: torch.Size, diagonal: int = 0):
        self.diagonal = diagonal
        self._input_ndim = len(input_shape)
        output_shape = _meta_output_shape(
            lambda x: x.diag(diagonal), input_shape)
        super().__init__(input_shape, output_shape, flags=LinearOpFlags.IS_NON_NEGATIVE | LinearOpFlags.IS_PURE_EXPANDING | LinearOpFlags.IS_PURE_CONTRACTING)

    def forward(self, x):
        return x.diag(self.diagonal)

    def backward(self, grad):
        if self._input_ndim == 1:
            # Forward was 1D→2D (create diagonal matrix); adjoint extracts diagonal
            return grad.diag(self.diagonal)
        else:
            # Forward was 2D→1D (extract diagonal); adjoint embeds into zeros
            result = torch.zeros(
                self.input_shape, dtype=grad.dtype, device=grad.device)
            n = len(grad)
            idx = torch.arange(n, device=grad.device)
            if self.diagonal >= 0:
                result[idx, idx + self.diagonal] = grad
            else:
                result[idx - self.diagonal, idx] = grad
            return result

    def __str__(self):
        return f"<diag {self.diagonal}>"
