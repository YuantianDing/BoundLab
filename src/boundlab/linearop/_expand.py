"""Expand LinearOp implemented through sparse EinsumOp."""

import torch

from boundlab.linearop._compat import ScalarOp


class ExpandOp:
    def __new__(cls, input_shape: torch.Size, output_shape: torch.Size):
        input_shape = torch.Size(input_shape)
        output_shape = torch.Size(output_shape)
        n_new = len(output_shape) - len(input_shape)
        assert n_new >= 0, (
            f"ExpandOp: output cannot have fewer dims than input ({len(output_shape)} < {len(input_shape)})"
        )
        if n_new == 0 and input_shape == output_shape:
            return ScalarOp(1.0, input_shape)
        if n_new == 0:
            return _make_expand_einsum(input_shape, output_shape)

        from boundlab.linearop._reshape import UnsqueezeOp

        padded_input = torch.Size([1] * n_new + list(input_shape))
        op = _make_expand_einsum(padded_input, output_shape)
        cur_shape = padded_input
        for _ in range(n_new):
            prev_shape = torch.Size(list(cur_shape)[1:])
            op = op @ UnsqueezeOp(prev_shape, dim=0)
            cur_shape = prev_shape
        return op


def _make_expand_einsum(input_shape: torch.Size, output_shape: torch.Size):
    from boundlab.linearop._einsum import EinsumOp

    tensor = torch.tensor(1.0).expand(*output_shape)
    output_dims = list(range(len(output_shape)))
    input_dims = []
    for d in range(len(input_shape)):
        if input_shape[d] == output_shape[d]:
            input_dims.append(d)
        else:
            assert input_shape[d] == 1, (
                f"ExpandOp: dim {d} has input size {input_shape[d]} != 1 and != output size {output_shape[d]}"
            )
            new_dim = tensor.dim()
            tensor = tensor.unsqueeze(new_dim)
            input_dims.append(new_dim)
    return EinsumOp(tensor, input_dims, output_dims, name=f"<expand {list(input_shape)} -> {list(output_shape)}>")
