
1. ~~Add `Conv2d` into the interpreter, using `EinsumOp` to implement `Conv2d` with `kernel_size == stride`.~~ Done — `_onnx_conv` in `src/boundlab/interp/__init__.py` handles `kernel_size == stride` via a 6-D reshape + `_onnx_einsum`.
2. ~~Enable swapping `EinsumOp` with ops like `GetSlicesOp` and `SetSlicesOp` when the swapping and reduce the size of `EinsumOp`.~~ Done — `GetSliceOp.__matmul__(EinsumOp)` and `SetSliceOp.__rmatmul__(EinsumOp)` in `src/boundlab/linearop/_indices.py` fuse by slicing the einsum tensor along the corresponding output-/input-side tensor dims (bail on mul_dim partial slices). Tests in `tests/test_einsum_slice_fusion.py`.
3. `vit/pruning.py` Implement differential verification on `vit_threshold.py`, leaving `heaviside_pruning` as a future work.
4. Linearize `heaviside_pruning((_, sy, _), (x, y, d))`:
   1. Let `(x, y, d) -> (x, y - h(-sy) * y, d + h(-sy) * y)`, Only `h(-sy) * y` need to be linearized.
   2. To linearize something like `h(s) * x` on `ls ≤ s ≤ us, lx ≤ x ≤ ux`, first, if `ls + us > 0`, we linearize `x - h(-s) * x` instead. Similarly, if `lx + ux > 0`, we linearize `-h(s)(-x)` instead. This ensures `ls + us <= 0` and `lx + ux <= 0` always holds, then we move on the cases:
      1. If `us < 0`, then `h(-s) * x` is always `0`, so we can linearize it as `0`.
      2. If `ls < 0 < us and ux ≤ 0`, let `lam = max(lx / (-ls), ux / us)` and provide two bounds: `lx ≤ h(s) * x - lam * s ≤ 0`.
      3. If `ls < 0 < us and ux > 0`, let `lam = min(ux / (ux - lx), (-lx) / (ux - lx))` and provide two bounds: `(1 - lam) * lx ≤ h(s) * x - lam * x ≤ (1 - lam) * ux`.
   3. Please try to fix the above logic if there are any mistakes, and implement it in `boundlab.diff.zono3`.