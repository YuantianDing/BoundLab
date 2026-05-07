
Refactor `boundlab.linearop`.

1. Add tests for `coo.MultiCOOTensorSum`. Make sure `apply_multiplicative` works properly with more than 5 non-disjoint terms. Also make sure `coo.MultiCOOTensor.__add__` works. 
2. `LinearOp` is now a `coo.MultiCOOTensorSum`, please pretty print its `COOSparsify` information (concise form) in `__str__`. 
3. Add tests for LinearOp to check its use of `coo.MultiCOOTensorSum` is correct, for `__add__`, `__mul__`, `abs`, etc.
4. Implement `EinsumOp`, construct a `coo.MultiCOOTensor` for `EinsumOp`. Make use of `COOSparsify.md_eye`. For a single standalone dimension, add a `COOSparsify.md_eye` for that dimension. For diagonal dimension (`mul` dimensions), use `COOSparsify.md_eye`. If the input tensor has `stride=0` in certain dimension, remove that dimension when putting into `coo.MultiCOOTensor.tn`. This dimension still need to be added into `COOSparsify`, `tn` can have less dimension than `COOSparsify`'s input dimensions.

!!! DO NOT TOUCH UNRELATED CODE !!!
!!! DO NOT OVERCOMPLEX THE PROBLEM !!!