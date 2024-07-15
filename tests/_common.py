import functools
import itertools
import math
from enum import IntFlag
from typing import Tuple, Callable

from PyQt5.QtCore import QEvent, QTimer
from PyQt5.QtWidgets import QApplication
from pytestqt.qtbot import QtBot

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


def schedule(fn: Callable, delay: int = 100):
    QTimer.singleShot(delay, fn)


def schedule_quit(app: QApplication, delay: int = 100):
    schedule(app.quit, delay)


class KGDQtWrapper:
    def __init__(self, qtbot, app, capfd):
        self.bot = qtbot
        self.app = app
        self.capfd = capfd


class TimedOffscreenApplication:
    def __init__(self, wrapper: KGDQtWrapper, duration=1):
        self.wrapper = wrapper
        self.duration = round(1000 * duration)
        self.error = False

    def __enter__(self) -> Tuple[QApplication, QtBot]:
        # print("[kgd-debug]", f"{self.__class__.__name__}.__enter__")
        QTimer.singleShot(self.duration, self.__failure)
        return self.wrapper.app, self.wrapper.bot

    def __exit__(self, type, value, traceback):
        # print("[kgd-debug]", f"{self.__class__.__name__}.__enter__")
        if self.error:
            raise self.TimeOutException(
                f"Application did not finish in {self.duration / 1000:g} seconds"
            )

        oe = self.wrapper.capfd.readouterr()
        err = oe.err
        err = "\n".join(
            f">  {e}"
            for e in err.split("\n")
            if not e.startswith("This plugin does not support") and e
        )
        if len(err) > 0:
            print(oe.out)
            raise AssertionError(f"Error output is not empty:\n{err}")

    class TimeOutException(Exception):
        pass

    def __failure(self, *_):
        # print("[kgd-debug]", "Failure!")
        self.wrapper.app.quit()
        self.error = True
