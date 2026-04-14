
1. Add `Conv2d` into the interpreter, using `EinsumOp` to implement `Conv2d` with `kernel_size == stride`.
2. Enable swapping `EinsumOp` with ops like `GetSlicesOp` and `SetSlicesOp` when the swapping and reduce the size of `EinsumOp`.
3. 