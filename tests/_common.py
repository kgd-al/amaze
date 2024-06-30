import functools
import itertools
import math
from enum import IntFlag

from amaze import StartLocation, Sign


class TestSize(IntFlag):
    __test__ = False
    SMALL = 1
    NORMAL = 2


@functools.lru_cache(maxsize=1)
def generate_large_maze_data_sample(size: TestSize):
    small = size is TestSize.SMALL

    def signs(value: float, _small):
        none = []
        one = [Sign(value=value)]
        two = [Sign(value=value), Sign("point", value)]
        return [none, one] if _small else [none, one, two]

    data = dict(
        width=[5],  # if small else [5, 10],
        height=[5],  # if small else [5, 10],
        seed=[10] if small else [0, 10],
        start=[StartLocation.SOUTH_WEST, StartLocation.NORTH_EAST],
        rotated=[True, False],
        unicursive=[True, False],
        p_lure=[0.5] if small else [0, 0.5],
        p_trap=[0, 0.5],
        clue=signs(1, small),
        lure=signs(0.25, small),
        trap=signs(0.5, False),
    )

    count = math.prod(len(d) for d in data.values())
    counts = str(count) + " = " + " * ".join(f"{len(d):{len(k)}}" for k, d in data.items())
    print("-" * len(counts))
    print("Normal scale mazes:", len(data), "fields")
    print("-" * (len(counts) // 2))
    print(" " * len(str(count)), " ", "   ".join(k for k in data))
    print(counts)
    print("Items")
    print("-" * len(counts))

    return [dict(zip(data.keys(), t)) for t in itertools.product(*data.values())]


MAZE_DATA_SAMPLE = [
    "M10_5x5_U",
    "M10_5x5_l.25_L.4_L.5",
    "M10_10x10_C1_l.2_L.25_t.5_T.5",
]
