

import torch

from boundlab.linearop import LinearOp




class ScatterOp(LinearOp):
    """A LinearOp that implements `torch.scatter` along a specified dimension with given indices."""
    # TODO
    pass

class GatherOp(LinearOp):
    """A LinearOp that implements `torch.gather` along a specified dimension with given indices."""
    # TODO
    pass

class ScatterIdxOp(LinearOp):
    
    def __init__(self, indices, input_shape: torch.Size, output_shape: torch.Size):
        self.indices = indices
        assert type(indices) == tuple and len(indices) == len(output_shape), "Indices must be a tuple of the same length as output_shape."
        for indices_i in indices:
            assert indices_i.shape == output_shape
        super().__init__(input_shape, output_shape)        
    
    def forward(self, x):
        result = torch.zeros(self.output_shape, device=x.device, dtype=x.dtype)
        result[self.indices] = x
        return result
    
    def backward(self, grad):
        return grad[self.indices]
    
    # TODO: implement other methods as needed.

class GatherIdxOp(LinearOp):
    
    def __init__(self, indices, input_shape: torch.Size, output_shape: torch.Size):
        self.indices = indices
        assert type(indices) == tuple and len(indices) == len(output_shape), "Indices must be a tuple of the same length as output_shape."
        for indices_i in indices:
            assert indices_i.shape == output_shape
        super().__init__(input_shape, output_shape)        
    
    def forward(self, x):
        return x[self.indices]
    
    def backward(self, grad):
        result = torch.zeros(self.input_shape, device=grad.device, dtype=grad.dtype)
        result[self.indices] = grad
        return result
    
    # TODO: implement other methods as needed.

class SliceOp(LinearOp):
    """A LinearOp that implements slicing e.g. ``tensor[3, 1:5, None]``."""
    # TODO
    pass

class SliceExpandOp(LinearOp):
    """A LinearOp that implements slicing with expansion. e.g.
    
    ```
    result = torch.zeros(5)
    result[None, 1:4, 3] = x
    ```
    """
    def __init__(self, indices, input_shape: torch.Size, output_shape: torch.Size):
        self.indices = indices
        assert type(indices) == tuple
        for (indices_i, output_shape_i) in zip(indices, output_shape):
            if isinstance(indices_i, slice):
                assert indices_i.stop <= output_shape_i, "Slice stop must be less than or equal to output shape."
            elif indices_i is None:
                assert output_shape_i == 1, "Output shape must be 1 for None indices."
            else:
                assert 0 <= indices_i < output_shape_i, "Index must be within the bounds of the output shape."
        super().__init__(input_shape, output_shape)

    # TODO
    pass