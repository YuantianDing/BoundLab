from __future__ import annotations

import dataclasses

import torch

from boundlab import expr
from boundlab.expr._core import Expr


@dataclasses.dataclass
class DiffExpr2:
    """A pair of expressions ``(x, y)`` for two-network differential tracking.

    All linear operators apply element-wise to both components.
    """

    x: Expr
    y: Expr
    
    def __post_init__(self):
        assert isinstance(self.x, Expr) and isinstance(self.y, Expr), "DiffExpr2 components must be Expr instances"

    @property
    def shape(self) -> torch.Size:
        return self.x.shape

    def _map_all(self, fn):
        return DiffExpr2(fn(self.x), fn(self.y))
    
    def get_const(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        x = self.x.get_const()
        if x is not None:
            y = self.y.get_const()
            if y is not None:
                return x, y
        return None

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------

    def __add__(self, other):
        if isinstance(other, int) and other == 0:
            return self
        if isinstance(other, (torch.Tensor, Expr)):
            return DiffExpr2(self.x + other, self.y + other)
        if isinstance(other, DiffExpr2):
            return DiffExpr2(self.x + other.x, self.y + other.y)
        return NotImplemented

    def __radd__(self, other):
        if isinstance(other, int) and other == 0:
            return self
        if isinstance(other, (torch.Tensor, Expr)):
            return DiffExpr2(other + self.x, other + self.y)
        if isinstance(other, DiffExpr3):
            return DiffExpr3(
                self.x + other.x,
                self.y + other.y,
                (self.x - self.y) + other.diff,
            )
        return NotImplemented

    def __neg__(self):
        return DiffExpr2(-self.x, -self.y)

    def __sub__(self, other):
        if isinstance(other, (torch.Tensor, Expr)):
            return DiffExpr2(self.x - other, self.y - other)
        if isinstance(other, DiffExpr2):
            return DiffExpr2(self.x - other.x, self.y - other.y)
        if isinstance(other, DiffExpr3):
            return DiffExpr3(
                self.x - other.x,
                self.y - other.y,
                (self.x - self.y) - other.diff,
            )
        return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, (torch.Tensor, Expr)):
            return DiffExpr2(other - self.x, other - self.y)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, (int, float, torch.Tensor)):
            return DiffExpr2(self.x * other, self.y * other)
        if isinstance(other, Expr):
            if (t := other.get_const()) is not None:
                return DiffExpr2(self.x * t, self.y * t)
            elif (tensors := self.get_const()) is not None:
                return DiffExpr2(other * tensors[0], other * tensors[1])
        if isinstance(other, DiffExpr2):
            if (tensors := self.get_const()) is not None:
                return DiffExpr2(other.x * tensors[0], other.y * tensors[1])
            elif (tensors := other.get_const()) is not None:
                return DiffExpr2(self.x * tensors[0], self.y * tensors[1])
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (int, float, torch.Tensor)):
            return DiffExpr2(other * self.x, other * self.y)
        if isinstance(other, Expr):
            return self.__mul__(other)
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, (int, float, torch.Tensor)):
            return DiffExpr2(self.x / other, self.y / other)
        if isinstance(other, DiffExpr2):
            if (tensors := other.get_const()) is not None:
                return DiffExpr2(self.x / tensors[0], self.y / tensors[1])
        return NotImplemented

    def __matmul__(self, other):
        if isinstance(other, torch.Tensor):
            return DiffExpr2(self.x @ other, self.y @ other)
        if isinstance(other, Expr):
            if (t := other.get_const()) is not None:
                return DiffExpr2(self.x @ t, self.y @ t)
            elif (tensors := self.get_const()) is not None:
                return DiffExpr2(tensors[0] @ other, tensors[1] @ other)
        if isinstance(other, DiffExpr2):
            if (tensors := self.get_const()) is not None:
                return DiffExpr2(tensors[0] @ other.x, tensors[1] @ other.y)
            elif (tensors := other.get_const()) is not None:
                return DiffExpr2(self.x @ tensors[0], self.y @ tensors[1])
        return NotImplemented

    def __rmatmul__(self, other):
        if isinstance(other, torch.Tensor):
            return DiffExpr2(other @ self.x, other @ self.y)
        if isinstance(other, Expr):
            if (t := other.get_const()) is not None:
                return DiffExpr2(t @ self.x, t @ self.y)
            elif (tensors := self.get_const()) is not None:
                return DiffExpr2(other @ tensors[0], other @ tensors[1])
        # if isinstance(other, DiffExpr3):
        #     # other is input triple (x, y, d); self is constant weight pair (W1, W2)
        #     # x@W1 − y@W2 = d@W1 + y@(W1−W2)
        #     wx, wy = _const_value(self.x), _const_value(self.y)
        #     if wx is not None and wy is not None:
        #         return DiffExpr3(
        #             other.x @ wx,
        #             other.y @ wy,
        #             other.diff @ wx + other.y @ (wx - wy),
        #         )
        return NotImplemented

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def __getitem__(self, indices):
        return DiffExpr2(self.x[indices], self.y[indices])

    def scatter(self, indices, output_shape):
        return self._map_all(lambda e: e.scatter(indices, output_shape))

    def gather(self, indices):
        return self._map_all(lambda e: e.gather(indices))

    # ------------------------------------------------------------------
    # Shape ops
    # ------------------------------------------------------------------

    def reshape(self, *shape):
        return self._map_all(lambda e: e.reshape(*shape))

    def permute(self, *dims):
        return self._map_all(lambda e: e.permute(*dims))

    def transpose(self, dim0, dim1):
        return self._map_all(lambda e: e.transpose(dim0, dim1))

    def flatten(self, start_dim=0, end_dim=-1):
        return self._map_all(lambda e: e.flatten(start_dim, end_dim))

    def unflatten(self, dim, sizes):
        return self._map_all(lambda e: e.unflatten(dim, sizes))

    def squeeze(self, dim=None):
        return self._map_all(lambda e: e.squeeze(dim))

    def unsqueeze(self, dim):
        return self._map_all(lambda e: e.unsqueeze(dim))

    def narrow(self, dim, start, length):
        return self._map_all(lambda e: e.narrow(dim, start, length))

    def expand(self, *sizes):
        return self._map_all(lambda e: e.expand(*sizes))

    def repeat(self, *sizes):
        return self._map_all(lambda e: e.repeat(*sizes))

    def tile(self, *sizes):
        return self._map_all(lambda e: e.tile(*sizes))

    def flip(self, dims):
        return self._map_all(lambda e: e.flip(dims))

    def roll(self, shifts, dims):
        return self._map_all(lambda e: e.roll(shifts, dims))

    def diag(self, diagonal=0):
        return self._map_all(lambda e: e.diag(diagonal))

    def mean(self, dim=None, keepdim=False):
        return self._map_all(lambda e: e.mean(dim=dim, keepdim=keepdim))
    
    def sum(self, dim=None, keepdim=False):
        return self._map_all(lambda e: e.sum(dim=dim, keepdim=keepdim))
    
    def __repr__(self):
        return f"DiffExpr2(x={self.x.bound_width().max().item()}, y={self.y.bound_width().max().item()})"


@dataclasses.dataclass
class DiffExpr3:
    """A triple ``(x, y, diff)`` for differential zonotope verification.

    ``x`` and ``y`` track each network's activations independently.
    ``diff`` over-approximates ``f₁(x) − f₂(y)``.

    For affine operations ``f(z) = W z + b``:
      - ``x``  and ``y``  receive both weight and bias.
      - ``diff`` receives only the weight (bias cancels: ``(Wx+b)−(Wy+b) = W(x−y)``).

    For pure-linear operations (no bias), all three components are updated
    identically.
    """

    x: Expr
    y: Expr
    diff: Expr
    def __post_init__(self):
        if isinstance(self.x, torch.Tensor):
            self.x = expr.ConstVal(self.x)
        if isinstance(self.y, torch.Tensor):
            self.y = expr.ConstVal(self.y)
        if isinstance(self.diff, torch.Tensor):
            self.diff = expr.ConstVal(self.diff)
        assert isinstance(self.x, Expr) and isinstance(self.y, Expr), "DiffExpr2 components must be Expr instances"
        assert isinstance(self.diff, Expr), "DiffExpr3 diff component must be an Expr instance"

    @property
    def shape(self) -> torch.Size:
        return self.x.shape

    def _map_all(self, fn):
        """Apply *fn* to all three components (pure-linear ops)."""
        return DiffExpr3(fn(self.x), fn(self.y), fn(self.diff))

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------

    def __add__(self, other):
        if isinstance(other, int) and other == 0:
            return self
        if isinstance(other, (torch.Tensor, Expr)):
            # Constant bias cancels in the diff component.
            return DiffExpr3(self.x + other, self.y + other, self.diff)
        if isinstance(other, DiffExpr2):
            return DiffExpr3(
                self.x + other.x,
                self.y + other.y,
                self.diff + (other.x - other.y),
            )
        if isinstance(other, DiffExpr3):
            return DiffExpr3(self.x + other.x, self.y + other.y, self.diff + other.diff)
        return NotImplemented

    def __radd__(self, other):
        if isinstance(other, int) and other == 0:
            return self
        if isinstance(other, (torch.Tensor, Expr)):
            return DiffExpr3(other + self.x, other + self.y, self.diff)
        if isinstance(other, DiffExpr2):
            return DiffExpr3(
                other.x + self.x,
                other.y + self.y,
                (other.x - other.y) + self.diff,
            )
        return NotImplemented

    def __neg__(self):
        return DiffExpr3(-self.x, -self.y, -self.diff)

    def __sub__(self, other):
        if isinstance(other, int) and other == 0:
            return self
        if isinstance(other, (torch.Tensor, Expr)):
            return DiffExpr3(self.x - other, self.y - other, self.diff)
        if isinstance(other, DiffExpr2):
            return DiffExpr3(
                self.x - other.x,
                self.y - other.y,
                self.diff - (other.x - other.y),
            )
        if isinstance(other, DiffExpr3):
            return DiffExpr3(self.x - other.x, self.y - other.y, self.diff - other.diff)
        return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, (torch.Tensor, Expr)):
            # ``other − (x, y, d)`` negates all three then adds constant to x/y.
            return DiffExpr3(other - self.x, other - self.y, -self.diff)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, (int, float, torch.Tensor)):
            return self._map_all(lambda e: e * other)
        if isinstance(other, Expr):
            if (v := other.get_const()) is not None:
                return self._map_all(lambda e: e * v)
        if isinstance(other, DiffExpr2):
            if (tensors := other.get_const()) is not None:
                # Bilinear diff identity: x*vx − y*vy = diff*vx + y*(vx − vy)
                return DiffExpr3(
                    self.x * tensors[0],
                    self.y * tensors[1],
                    self.diff * tensors[0] + self.y * (tensors[0] - tensors[1]),
                )
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (int, float, torch.Tensor)):
            return self._map_all(lambda e: other * e)
        if isinstance(other, Expr):
            if (v := other.get_const()) is not None:
                return self._map_all(lambda e: v * e)
        if isinstance(other, DiffExpr2):
            if (tensors := other.get_const()) is not None:
                return DiffExpr3(
                    tensors[0] * self.x,
                    tensors[1] * self.y,
                    tensors[0] * self.diff + (tensors[0] - tensors[1]) * self.y,
                )
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, (int, float, torch.Tensor)):
            return self._map_all(lambda e: e / other)
        if isinstance(other, DiffExpr2):
            if (tensors := other.get_const()) is not None:
                vx, vy = tensors
                if torch.allclose(vx, vy):
                    return self._map_all(lambda e: e / vx)
                return DiffExpr3(
                    self.x / vx,
                    self.y / vy,
                    self.diff / vx + self.y * (1.0 / vx - 1.0 / vy),
                )
        return NotImplemented

    def __matmul__(self, other):
        if isinstance(other, torch.Tensor):
            return self._map_all(lambda e: e @ other)
        if isinstance(other, Expr):
            if (v := other.get_const()) is not None:
                return self._map_all(lambda e: e @ v)
        if isinstance(other, DiffExpr2):
            # self is input triple (x, y, d); other is constant weight pair (W1, W2)
            # x@W1 − y@W2 = d@W1 + y@(W1−W2)
            if (tensors := other.get_const()) is not None:
                return DiffExpr3(
                    self.x @ tensors[0],
                    self.y @ tensors[1],
                    self.diff @ tensors[0] + self.y @ (tensors[0] - tensors[1]),
                )
        return NotImplemented

    def __rmatmul__(self, other):
        if isinstance(other, torch.Tensor):
            return self._map_all(lambda e: other @ e)
        if isinstance(other, Expr):
            if (v := other.get_const()) is not None:
                return self._map_all(lambda e: v @ e)
        if isinstance(other, DiffExpr2):
            # other is constant weight pair (W1, W2); self is input triple (x, y, d)
            # W1@x − W2@y = W1@d + (W1−W2)@y
            if (tensors := other.get_const()) is not None:
                return DiffExpr3(
                    tensors[0] @ self.x,
                    tensors[1] @ self.y,
                    tensors[0] @ self.diff + (tensors[0] - tensors[1]) @ self.y,
                )
        return NotImplemented

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def __getitem__(self, indices):
        # Plain int: tuple-unpacking (e.g. from getitem nodes in torch.export graph).
        if isinstance(indices, int):
            return (self.x, self.y, self.diff)[indices]
        return self._map_all(lambda e: e[indices])

    def scatter(self, indices, output_shape):
        return self._map_all(lambda e: e.scatter(indices, output_shape))

    def gather(self, indices):
        return self._map_all(lambda e: e.gather(indices))

    # ------------------------------------------------------------------
    # Shape ops
    # ------------------------------------------------------------------

    def reshape(self, *shape):
        return self._map_all(lambda e: e.reshape(*shape))

    def permute(self, *dims):
        return self._map_all(lambda e: e.permute(*dims))

    def transpose(self, dim0, dim1):
        return self._map_all(lambda e: e.transpose(dim0, dim1))

    def flatten(self, start_dim=0, end_dim=-1):
        return self._map_all(lambda e: e.flatten(start_dim, end_dim))

    def unflatten(self, dim, sizes):
        return self._map_all(lambda e: e.unflatten(dim, sizes))

    def squeeze(self, dim=None):
        return self._map_all(lambda e: e.squeeze(dim))

    def unsqueeze(self, dim):
        return self._map_all(lambda e: e.unsqueeze(dim))

    def narrow(self, dim, start, length):
        return self._map_all(lambda e: e.narrow(dim, start, length))

    def expand(self, *sizes):
        return self._map_all(lambda e: e.expand(*sizes))

    def repeat(self, *sizes):
        return self._map_all(lambda e: e.repeat(*sizes))

    def tile(self, *sizes):
        return self._map_all(lambda e: e.tile(*sizes))

    def flip(self, dims):
        return self._map_all(lambda e: e.flip(dims))

    def roll(self, shifts, dims):
        return self._map_all(lambda e: e.roll(shifts, dims))

    def diag(self, diagonal=0):
        return self._map_all(lambda e: e.diag(diagonal))

    def mean(self, dim=None, keepdim=False):
        return self._map_all(lambda e: e.mean(dim=dim, keepdim=keepdim))
    
    def sum(self, dim=None, keepdim=False):
        return self._map_all(lambda e: e.sum(dim=dim, keepdim=keepdim))
    
    def __str__(self):
        X = str(self.x).replace("\n", "\n    ")
        Y = str(self.y).replace("\n", "\n    ")
        D = str(self.diff).replace("\n", "\n    ")
        return f"DiffExpr3 {{\n    x: {X},\n    y: {Y},\n    diff: {D}\n}}"
    
    def __repr__(self):
        return f"DiffExpr3(x={self.x.bound_width().max().item()}, y={self.y.bound_width().max().item()}, diff={self.diff.bound_width().max().item()})"

__all__ = ["DiffExpr2", "DiffExpr3"]