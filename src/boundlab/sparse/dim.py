


from typing import Any


class Dim:
    def __init__(self, length: int, ordering: float, name: str | None = None):
        self.length = length
        self.ordering = ordering
        self.name = name

    def _sort_key(self) -> tuple[float, int, str, int]:
        return (self.ordering, self.length, self.name or "", id(self))
    
    def __le__(self, other: "Dim") -> bool:
        return self._sort_key() <= other._sort_key()
    def __lt__(self, other: "Dim") -> bool:
        return self._sort_key() < other._sort_key()
    def __eq__(self, other: Any) -> bool:
        return self is other
    def __hash__(self) -> int:
        return id(self)
    def __str__(self):
        return self.name if self.name else hex(id(self))
    def __repr__(self):
        return str(self)

    def clone(self, name: str | None = None) -> "Dim":
        return Dim(length=self.length, ordering=self.ordering, name=name if name is not None else self.name)
