
import dataclasses

import torch

from boundlab import expr
from boundlab.expr._core import Expr


def _const_value(e: Expr) -> torch.Tensor | None:
    """Return the concrete tensor if *e* is a pure constant expression, else None.

    Works for :class:`~boundlab.expr.ConstVal` and any :class:`~boundlab.expr.AffineSum`
    that has no symbolic children.
    """
    assert isinstance(e, expr.AffineSum), "Expected an AffineSum or ConstVal"
    children = getattr(e, "children", ())
    constant = getattr(e, "constant", None)
    if not children and constant is not None:
        return constant
    return None


@dataclasses.dataclass
class DiffExpr2:
    """A pair of expressions ``(x, y)`` for two-network differential tracking.

    All linear operators apply element-wise to both components.
    """

    x: Expr
    y: Expr

    @property
    def shape(self) -> torch.Size:
        return self.x.shape

    def _map(self, fn):
        return DiffExpr2(fn(self.x), fn(self.y))

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
        return NotImplemented

    def __neg__(self):
        return DiffExpr2(-self.x, -self.y)

    def __sub__(self, other):
        if isinstance(other, (torch.Tensor, Expr)):
            return DiffExpr2(self.x - other, self.y - other)
        if isinstance(other, DiffExpr2):
            return DiffExpr2(self.x - other.x, self.y - other.y)
        return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, (torch.Tensor, Expr)):
            return DiffExpr2(other - self.x, other - self.y)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, (int, float, torch.Tensor)):
            return DiffExpr2(self.x * other, self.y * other)
        if isinstance(other, Expr):
            v = _const_value(other)
            if v is not None:
                return DiffExpr2(self.x * v, self.y * v)
        if isinstance(other, DiffExpr2):
            vx, vy = _const_value(other.x), _const_value(other.y)
            if vx is not None and vy is not None:
                return DiffExpr2(self.x * vx, self.y * vy)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (int, float, torch.Tensor)):
            return DiffExpr2(other * self.x, other * self.y)
        if isinstance(other, Expr):
            v = _const_value(other)
            if v is not None:
                return DiffExpr2(v * self.x, v * self.y)
        if isinstance(other, DiffExpr2):
            vx, vy = _const_value(other.x), _const_value(other.y)
            if vx is not None and vy is not None:
                return DiffExpr2(vx * self.x, vy * self.y)
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, (int, float, torch.Tensor)):
            return DiffExpr2(self.x / other, self.y / other)
        return NotImplemented

    def __matmul__(self, other):
        if isinstance(other, torch.Tensor):
            return DiffExpr2(self.x @ other, self.y @ other)
        if isinstance(other, Expr):
            v = _const_value(other)
            if v is not None:
                return DiffExpr2(self.x @ v, self.y @ v)
        if isinstance(other, DiffExpr3):
            # self is constant weight pair (W1, W2); other is input triple (x, y, d)
            # W1@x − W2@y = W1@d + (W1−W2)@y
            wx, wy = _const_value(self.x), _const_value(self.y)
            if wx is not None and wy is not None:
                return DiffExpr3(
                    wx @ other.x,
                    wy @ other.y,
                    wx @ other.diff + (wx - wy) @ other.y,
                )
        return NotImplemented

    def __rmatmul__(self, other):
        if isinstance(other, torch.Tensor):
            return DiffExpr2(other @ self.x, other @ self.y)
        if isinstance(other, Expr):
            v = _const_value(other)
            if v is not None:
                return DiffExpr2(v @ self.x, v @ self.y)
        if isinstance(other, DiffExpr3):
            # other is input triple (x, y, d); self is constant weight pair (W1, W2)
            # x@W1 − y@W2 = d@W1 + y@(W1−W2)
            wx, wy = _const_value(self.x), _const_value(self.y)
            if wx is not None and wy is not None:
                return DiffExpr3(
                    other.x @ wx,
                    other.y @ wy,
                    other.diff @ wx + other.y @ (wx - wy),
                )
        return NotImplemented

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def __getitem__(self, indices):
        return DiffExpr2(self.x[indices], self.y[indices])

    def scatter(self, indices, output_shape):
        return self._map(lambda e: e.scatter(indices, output_shape))

    def gather(self, indices):
        return self._map(lambda e: e.gather(indices))

    # ------------------------------------------------------------------
    # Shape ops
    # ------------------------------------------------------------------

    def reshape(self, *shape):
        return self._map(lambda e: e.reshape(*shape))

    def permute(self, *dims):
        return self._map(lambda e: e.permute(*dims))

    def transpose(self, dim0, dim1):
        return self._map(lambda e: e.transpose(dim0, dim1))

    def flatten(self, start_dim=0, end_dim=-1):
        return self._map(lambda e: e.flatten(start_dim, end_dim))

    def unflatten(self, dim, sizes):
        return self._map(lambda e: e.unflatten(dim, sizes))

    def squeeze(self, dim=None):
        return self._map(lambda e: e.squeeze(dim))

    def unsqueeze(self, dim):
        return self._map(lambda e: e.unsqueeze(dim))

    def narrow(self, dim, start, length):
        return self._map(lambda e: e.narrow(dim, start, length))

    def expand(self, *sizes):
        return self._map(lambda e: e.expand(*sizes))

    def repeat(self, *sizes):
        return self._map(lambda e: e.repeat(*sizes))

    def tile(self, *sizes):
        return self._map(lambda e: e.tile(*sizes))

    def flip(self, dims):
        return self._map(lambda e: e.flip(dims))

    def roll(self, shifts, dims):
        return self._map(lambda e: e.roll(shifts, dims))

    def diag(self, diagonal=0):
        return self._map(lambda e: e.diag(diagonal))


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
            v = _const_value(other)
            if v is not None:
                return self._map_all(lambda e: e * v)
        if isinstance(other, DiffExpr2):
            vx, vy = _const_value(other.x), _const_value(other.y)
            if vx is not None and vy is not None:
                # Bilinear diff identity: x*vx − y*vy = diff*vx + y*(vx − vy)
                return DiffExpr3(
                    self.x * vx,
                    self.y * vy,
                    self.diff * vx + self.y * (vx - vy),
                )
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (int, float, torch.Tensor)):
            return self._map_all(lambda e: other * e)
        if isinstance(other, Expr):
            v = _const_value(other)
            if v is not None:
                return self._map_all(lambda e: v * e)
        if isinstance(other, DiffExpr2):
            vx, vy = _const_value(other.x), _const_value(other.y)
            if vx is not None and vy is not None:
                return DiffExpr3(
                    vx * self.x,
                    vy * self.y,
                    vx * self.diff + (vx - vy) * self.y,
                )
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, (int, float, torch.Tensor)):
            return self._map_all(lambda e: e / other)
        return NotImplemented

    def __matmul__(self, other):
        if isinstance(other, torch.Tensor):
            return self._map_all(lambda e: e @ other)
        if isinstance(other, Expr):
            v = _const_value(other)
            if v is not None:
                return self._map_all(lambda e: e @ v)
        if isinstance(other, DiffExpr2):
            # self is input triple (x, y, d); other is constant weight pair (W1, W2)
            # x@W1 − y@W2 = d@W1 + y@(W1−W2)
            wx, wy = _const_value(other.x), _const_value(other.y)
            if wx is not None and wy is not None:
                return DiffExpr3(
                    self.x @ wx,
                    self.y @ wy,
                    self.diff @ wx + self.y @ (wx - wy),
                )
        return NotImplemented

    def __rmatmul__(self, other):
        if isinstance(other, torch.Tensor):
            return self._map_all(lambda e: other @ e)
        if isinstance(other, Expr):
            v = _const_value(other)
            if v is not None:
                return self._map_all(lambda e: v @ e)
        if isinstance(other, DiffExpr2):
            # other is constant weight pair (W1, W2); self is input triple (x, y, d)
            # W1@x − W2@y = W1@d + (W1−W2)@y
            wx, wy = _const_value(other.x), _const_value(other.y)
            if wx is not None and wy is not None:
                return DiffExpr3(
                    wx @ self.x,
                    wy @ self.y,
                    wx @ self.diff + (wx - wy) @ self.y,
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

__all__ = ["DiffExpr2", "DiffExpr3"]