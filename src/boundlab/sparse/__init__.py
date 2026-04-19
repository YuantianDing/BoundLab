


from . import coos, factors, ops, table

__all__ = ["coos", "factors", "ops", "table"]

    


# class SparseGroup:
#     def __init__(self, dims: list[int], indices: list[torch.Tensor], output_shape: list[int]):
#         sorted = sorted([(dim, idx, output_shape) for dim, idx, output_shape in zip(dims, indices, output_shape)], key=lambda x: x[0])
#         self.dims = [r[0] for r in sorted]
#         self.indices = [r[1] for r in sorted]
#         self.output_shape = [r[2] for r in sorted]


    



    

    
