"""
Generic coordinate helpers

Contains 2D vector (and points) and cell-aligned positions
"""

import math
from typing import Tuple


class Vec:
    """ Generic 2d vector with limited arithmetic operations """
    def __init__(self, x: float, y: float):
        self.x, self.y = x, y

    def __repr__(self): return f"({self.x}, {self.y})"
    def __str__(self): return f"{self.x:.2g}, {self.y:.2g}"

    def __bool__(self): return bool(self.x != 0 or self.y != 0)

    def __eq__(self, other):
        try:
            return self.x == other[0] and self.y == other[1]
        except:
            return False

    def __iadd__(self, other):
        self.x += other[0]
        self.y += other[1]
        return self

    def __add__(self, other): return type(self)(self.x + other.x, self.y + other.y)
    def __sub__(self, other): return type(self)(self.x - other.x, self.y - other.y)

    def __mul__(self, k): return type(self)(self.x * k, self.y * k)
    def __rmul__(self, k): return self.__mul__(k)

    def __truediv__(self, k): return type(self)(self.x / k, self.y / k)

    def __getitem__(self, i):
        if i < 0 or 1 < i:
            raise IndexError
        return self.x if i == 0 else self.y

    def __iter__(self): return iter((self.x, self.y))

    def is_null(self): return bool(self)

    def length(self): return math.sqrt(self.x**2 + self.y**2)

    def copy(self) -> 'Vec': return type(self)(self.x, self.y)

    @classmethod
    def null(cls): return cls(0, 0)


AlignedPos = Tuple[int, int]


class Pos(Vec):
    """ A position inside the maze """
    def aligned(self) -> AlignedPos:
        return int(self.x), int(self.y)
