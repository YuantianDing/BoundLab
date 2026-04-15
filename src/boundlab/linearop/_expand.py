"""Expand LinearOp — implemented as EinsumOp for broadcast expansion."""

import torch

from boundlab.linearop._base import ScalarOp, ComposedOp


class ExpandOp:
    """Broadcast-expand dimensions.

    Implemented as an EinsumOp with a ones template tensor.
    When ``len(input_shape) < len(output_shape)``, leading size-1 dims
    are prepended to input_shape automatically.

    Uses ``torch.tensor(1.0).expand(output_shape)`` as the einsum tensor
    to save memory (stride-zero storage).
    """

    def __new__(cls, input_shape: torch.Size, output_shape: torch.Size):
        if not isinstance(input_shape, torch.Size):
            input_shape = torch.Size(input_shape)
        if not isinstance(output_shape, torch.Size):
            output_shape = torch.Size(output_shape)
        n_new = len(output_shape) - len(input_shape)
        assert n_new >= 0, \
            f"ExpandOp: output cannot have fewer dims than input ({len(output_shape)} < {len(input_shape)})"
        if n_new == 0 and input_shape == output_shape:
            return ScalarOp(1.0, input_shape)
        if n_new == 0:
            return _make_expand_einsum(input_shape, output_shape)
        # Prepend n_new size-1 dims then expand
        from boundlab.linearop._reshape import UnsqueezeOp
        padded_input = torch.Size([1] * n_new + list(input_shape))
        expand_op = _make_expand_einsum(padded_input, output_shape)
        # Compose: expand_op @ unsqueeze(0)^n_new
        # Each unsqueeze adds a dim at position 0, building up from input_shape to padded_input
        op = expand_op
        cur_shape = padded_input
        for i in range(n_new):
            # Remove the leading 1 to get intermediate shape
            prev_shape = torch.Size(list(cur_shape)[1:])
            unsq = UnsqueezeOp(prev_shape, dim=0)
            op = op @ unsq
            cur_shape = prev_shape
        return op


def _make_expand_einsum(input_shape: torch.Size, output_shape: torch.Size):
    """Build an EinsumOp that performs broadcast expansion."""
    from boundlab.linearop._einsum import EinsumOp

    tensor = torch.tensor(1.0).expand(*output_shape)
    output_dims = list(range(len(output_shape)))
    input_dims = []

    for d in range(len(input_shape)):
        if input_shape[d] == output_shape[d]:
            # Shared dim (mul_dim)
            input_dims.append(d)
        else:
            assert input_shape[d] == 1, \
                f"ExpandOp: dim {d} has input size {input_shape[d]} != 1 and != output size {output_shape[d]}"
            # Add a new size-1 tensor dim for this input dim
            new_dim = tensor.dim()
            tensor = tensor.unsqueeze(new_dim)
            input_dims.append(new_dim)

    return EinsumOp(tensor, input_dims, output_dims, name=f"<expand {list(input_shape)} -> {list(output_shape)}>")
