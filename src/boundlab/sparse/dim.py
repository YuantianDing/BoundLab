


from typing import Any


class Dim:
    def __init__(self, length: int, ordering: float, name: str | None = None):
        self.length = length
        self.ordering = ordering
        self.name = name
    
    def __le__(self, other: "Dim") -> bool:
        return self.ordering <= other.ordering
    def __lt__(self, other: "Dim") -> bool:
        return self.ordering < other.ordering
    def __eq__(self, other: Any) -> bool:
        return self is other
    def __hash__(self) -> int:
        return id(self)
    def __str__(self):
        return self.name if self.name else hex(id(self))
    def __repr__(self):
        return str(self)