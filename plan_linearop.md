
Split `_shape.py` into separate files for shape ops and indexing ops.

1. `_reshape.py`: Make `FlattenOp`, `UnflatternOp`, `SqueezeOp` and `UnsqueezeOp` subclass of `ReshapeOp`. Replace `sizes` with accurate `output_shape` and `input_shape` in constructors. `len(input_shape) == len(output_shape)` is enforced for simplicity. 

2. `_expand.py`: Make `ExpandOp` subclass of `EinsumOp`. Using `torch.tensor(1.0).expand(shape)` as the template for the einsum tensor to save memory. Initialize `ExpandOp` using `def __init__(self, input_shape: torch.Size, output_shape: torch.Size)`, where `len(input_shape) == len(output_shape)` is enforced for simplicity.

3. `_permute.py`: `PermuteOp` and `TransposeOp` are unchanged.

4. `_slicing.py`: `GetSliceOp(input_shape: torch.Size, slices: list[list[slice]])` (`len(input_shape) == len(slices)`), `SetSliceOp(output_shape: torch.Size, slices: list[list[slice]])` (`len(output_shape) == len(slices)`) are two classes that allow multiple slices along each dimension.

5. `_indexing.py`: `GetIndicesOp(input_shape: torch.Size, dim: int, indices: torch.Tensor, added_shape: torch.Size)` and `SetIndicesOp(output_shape: torch.Size, dim: int, indices: torch.Tensor, added_shape: torch.Size)` are added with a `dim` to specify the dimension along which to index. So, the output shape of `GetIndicesOp` is `input_shape[:dim] + added_shape + input_shape[dim+1:]`, and the input shape of `SetIndicesOp` is `output_shape[:dim] + added_shape + output_shape[dim+1:]`, where `indices` has shape `[input_shape[dim], len(added_shape)]`.

Implement fusion and swapping rules for `EinsumOp` with these new ops:

1. `ReshapeOp`/`PermuteOp` keep the current fusion rules with `EinsumOp` (i.e., `ReshapeOp @ EinsumOp` and `EinsumOp @ ReshapeOp` are both supported if the shapes are compatible).
2. Simplify `ExpandOp` fusion logic with `EinsumOp` after enforcing `len(input_shape) == len(output_shape)` in `ExpandOp`. `ExpandOp @ EinsumOp` and `EinsumOp @ ExpandOp` are both supported if the shapes are compatible, and the einsum tensor is sliced accordingly. No swapping rules are needed for `ExpandOp` since it can fuse with `EinsumOp` in both orders.
3. For `GetSliceOp @ EinsumOp` and `EinsumOp @ SetSliceOp`, implement fusion with `EinsumOp` when the slicing dimensions are all `dot`/`batch` dimensions. Otherwise, swapping is needed to move `GetSliceOp`/`SetSliceOp` past `EinsumOp`. For the `dot`/`batch` dimensions, the corresponding dimensions of the einsum tensor are sliced accordingly. For the `mul` dimensions, not only the einsum tensor need to be sliced in those dimensions, but an additional `GetSliceOp`/`SetSliceOp` with the same slicing pattern also needs to be inserted after/before the `EinsumOp` to slice the output/input of the `EinsumOp` in those dimensions.
4. For `GetIndicesOp @ EinsumOp` and `EinsumOp @ SetIndicesOp`, implement fusion with `EinsumOp` when the indexing dimension is a `dot`/`batch` dimension. Otherwise, swapping is needed to move `GetIndicesOp`/`SetIndicesOp` past `EinsumOp`. For the `dot`/`batch` dimension, the corresponding dimension of the einsum tensor is indexed accordingly. For the `mul` dimensions, not only the einsum tensor need to be indexed in that dimension, but an additional `GetIndicesOp`/`SetIndicesOp` with the same indexing pattern also needs to be inserted after/before the `EinsumOp` to index the output/input of the `EinsumOp` in that dimension.

Implement fusion rules within those ops when possible, e.g., `GetSliceOp @ GetSliceOp` can be fused into a single `GetSliceOp` with the combined slicing pattern.

Not that the reviewer won't check lengthy fusion/swapping logic line by line, but please make sure the logic is short and simple, as the operator specifications are simple.

Update `Expr` APIs to use the new ops, e.g., `expand_on()` should use `ExpandOp` now.

Fix all problem in the tests. Add more tests if necessary, especially for the new ops and their fusion/swapping with `EinsumOp`.