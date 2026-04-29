from functools import reduce
from operator import mul
from typing_extensions import Self


class Shape:
    """
    Helper class to manipulate shapes.
    """

    def __init__(self, shape: list[int]):
        self._shape = shape.copy()

    def to_list(self) -> list[int]:
        return self._shape.copy()

    def copy(self) -> Self:
        return self.__class__(self.to_list())

    @property
    def batch_axis(self) -> int:
        return 0

    @property
    def batch_size(self) -> int:
        return self._shape[self.batch_axis]

    @batch_size.setter
    def batch_size(self, new_size: int) -> int:
        self._shape.pop(self.batch_axis)
        self._shape.insert(self.batch_axis, new_size)

        return new_size

    @property
    def elem_size(self) -> int:
        shape_without_batch = self._shape.copy()
        shape_without_batch.pop(self.batch_axis)
        return reduce(mul, shape_without_batch, 1)
