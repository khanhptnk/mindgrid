from minigrid.core.world_object import WorldObj
from random import Random

import numpy as np
from enum import Enum
from typing import Union, List, Tuple


class DeterministicRandom(Random):
    def choice(self, x):
        try:
            x = sorted(x)
        except:
            pass
        return super().choice(x)

    def sample(self, x, k):
        try:
            x = sorted(x)
        except:
            pass
        return super().sample(x, k)


class CustomEnum(Enum):
    @classmethod
    def has_value(cls, value):
        if isinstance(value, Enum):
            return value in cls
        return any(value == item.name for item in cls)

def get_adjacent_cells(
    cell: Tuple[int, int], ret_as_list: bool = False
) -> Union[set, List]:
    x, y = cell
    adj_cells = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    if ret_as_list:
        return adj_cells
    return set(adj_cells)

def get_diagonally_adjacent_cells(cell: Tuple[int, int]) -> set:
    x, y = cell
    return set([(x + 1, y + 1), (x + 1, y - 1), (x - 1, y + 1), (x - 1, y - 1)])


def to_enum(enum: Enum, value: Union[List, str]) -> List:
    if isinstance(value, str):
        return enum[value]
    return [enum[val] for val in value]


