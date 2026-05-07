"""Slice-based indexing LinearOp implementations.

``GetSliceOp`` and ``SetSliceOp`` use a structured ``list[list[slice]]``
format where each dimension has a list of non-overlapping slices.
``len(input_shape) == len(slices)`` is enforced.
"""

import torch

from boundlab.linearop._base import LinearOp, LinearOpFlags, ComposedOp
from boundlab.utils import merge_name


class GetSliceOp(LinearOp):
    """Extract sliced regions from a tensor.

    Args:
        input_shape: Shape of the input tensor.
        slices: Per-dimension list of slices. ``len(slices) == len(input_shape)``.
    """

    def __init__(self, input_shape: torch.Size, slices: list[list[slice]]):
        assert len(input_shape) == len(slices), \
            f"len(input_shape)={len(input_shape)} != len(slices)={len(slices)}"
        self.slices = _normalize_slices(slices, input_shape)
        output_shape = torch.Size(_output_size(self.slices[d]) for d in range(len(input_shape)))
        # TODO

    def __str__(self):
        parts = []
        for dim_slices in self.slices:
            if len(dim_slices) == 1:
                s = dim_slices[0]
                parts.append(f"{s.start}:{s.stop}")
            else:
                parts.append("[" + ",".join(f"{s.start}:{s.stop}" for s in dim_slices) + "]")
        return f"<getslice {','.join(parts)}>"


class SetSliceOp(LinearOp):
    """Embed input into zeros at specified slice positions.

    Args:
        output_shape: Shape of the output tensor (zeros template).
        slices: Per-dimension list of slices. ``len(output_shape) == len(slices)``.
    """

    def __init__(self, output_shape: torch.Size, slices: list[list[slice]]):
        assert len(output_shape) == len(slices), \
            f"len(output_shape)={len(output_shape)} != len(slices)={len(slices)}"
        self.slices = _normalize_slices(slices, output_shape)
        input_shape = torch.Size(_output_size(self.slices[d]) for d in range(len(output_shape)))
        # TODO

    def __str__(self):
        parts = []
        for dim_slices in self.slices:
            if len(dim_slices) == 1:
                s = dim_slices[0]
                parts.append(f"{s.start}:{s.stop}")
            else:
                parts.append("[" + ",".join(f"{s.start}:{s.stop}" for s in dim_slices) + "]")
        return f"<setslice {','.join(parts)}>"

