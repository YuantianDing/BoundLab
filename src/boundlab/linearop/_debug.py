"""Debug-only Jacobian builders for LinearOp subclasses."""

import torch


def _numel(shape: torch.Size) -> int:
    result = 1
    for size in shape:
        result *= int(size)
    return result


def _zero_stride_axes(tensor: torch.Tensor, axes: tuple[int, ...] = ()) -> torch.Tensor:
    if tensor.ndim == 0:
        return tensor
    axes = tuple(axis if axis >= 0 else tensor.ndim + axis for axis in axes)
    strides = tuple(0 if idx in axes else stride for idx, stride in enumerate(tensor.stride()))
    return torch.as_strided(tensor, tensor.shape, strides)


def jacobian_from_function(input_shape: torch.Size, output_shape: torch.Size, fn, zero_stride_axes: tuple[int, ...] = ()) -> torch.Tensor:
    input_shape = torch.Size(input_shape)
    output_shape = torch.Size(output_shape)
    input_numel = _numel(input_shape)
    if input_numel == 0:
        return _zero_stride_axes(torch.empty(output_shape + input_shape), zero_stride_axes)

    basis = torch.eye(input_numel).reshape((input_numel,) + input_shape)
    outputs = torch.stack([fn(basis[i]) for i in range(input_numel)])
    outputs = outputs.reshape((input_numel,) + output_shape)
    result = outputs.permute(tuple(range(1, outputs.ndim)) + (0,)).reshape(output_shape + input_shape).contiguous()
    return _zero_stride_axes(result, zero_stride_axes)


def jacobian_from_einsum_coeff(
    coeff: torch.Tensor,
    input_axes: list[int],
    output_axes: list[int],
) -> torch.Tensor:
    input_shape = torch.Size(coeff.shape[axis] for axis in input_axes)
    output_shape = torch.Size(coeff.shape[axis] for axis in output_axes)
    dtype = torch.get_default_dtype() if coeff.dtype is torch.bool else coeff.dtype
    coeff = coeff.to(dtype)
    result = torch.zeros(output_shape + input_shape, dtype=dtype, device=coeff.device)
    if coeff.numel() == 0:
        return result

    coords = (
        torch.zeros((1, 0), dtype=torch.long, device=coeff.device)
        if coeff.ndim == 0
        else torch.cartesian_prod(*[torch.arange(size, device=coeff.device) for size in coeff.shape])
    )
    if coords.ndim == 1:
        coords = coords.unsqueeze(1)
    index = tuple(coords[:, axis] for axis in output_axes + input_axes)
    result.index_put_(index, coeff.reshape(-1), accumulate=True)
    return result.contiguous()
