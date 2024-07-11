import functools
import itertools
import math
from enum import IntFlag

from PyQt5.QtCore import QEvent, QObject
from PyQt5.QtWidgets import QApplication

from amaze import StartLocation, Sign
from amaze.visu import MainWindow


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
    counts = (
        str(count) + " = " + " * ".join(f"{len(d):{len(k)}}" for k, d in data.items())
    )
    print()
    print("-" * len(counts))
    print(size.name, "scale mazes:", len(data), "fields")
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


class EventTypes:  # pragma: no cover
    @staticmethod
    def _build():
        """Create mapping for all known event types."""
        string_names = {}
        for name in vars(QEvent):
            attribute = getattr(QEvent, name)
            if isinstance(attribute, QEvent.Type):
                string_names[attribute] = name
        return string_names

    _names = _build()

    @classmethod
    def as_string(cls, event: QEvent.Type) -> str:
        """Return the string name for this event."""
        try:
            return cls._names[event]
        except KeyError:
            return f"UnknownEvent:{event}"


class EventObserver(QObject):
    def __init__(self, app: QApplication):
        super().__init__()
        self._app = app
        self._app.installEventFilter(self)

    def eventFilter(self, obj, event):
        # print(EventTypes.as_string(event.type()), obj, event)
        if event.type() == QEvent.WindowActivate and isinstance(obj, MainWindow):
            obj.close()
            return False
        else:
            return super().eventFilter(obj, event)
