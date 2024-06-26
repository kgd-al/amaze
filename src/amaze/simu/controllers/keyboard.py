from abc import ABCMeta
from logging import getLogger
from typing import List
from zipfile import ZipFile

from PyQt5.QtCore import QObject, Qt, QEvent
from PyQt5.QtGui import QKeyEvent

from .base import BaseController, Robot, Vec, OutputType, InputType
from ...visu.widgets import qt_application

logger = getLogger(__name__)


class __Meta(type(QObject), ABCMeta):
    pass


class KeyboardController(QObject, BaseController, metaclass=__Meta):
    _savable = False

    def __init__(self, robot_data: Robot.BuildData):
        super().__init__(robot_data=robot_data)
        qt_application(allow_create=False).installEventFilter(self)
        self.actions = {
            Qt.Key_Right: Vec(1, 0),
            Qt.Key_Up: Vec(0, 1),
            Qt.Key_Left: Vec(-1, 0),
            Qt.Key_Down: Vec(0, -1),
        }

        if self.output_type == OutputType.DISCRETE:
            self.actions_queue = []
        else:
            self.down_keys = {a: False for a in self.actions.keys()}
            self.current_action = Vec.null()

    def eventFilter(self, obj: 'QObject', e: 'QEvent') -> bool:
        if not isinstance(e, QKeyEvent):
            return False
        e: QKeyEvent
        print(f"[kgd-debug] KeyEvent({e.type()}, {hex(e.key())})")
        if self.output_type is OutputType.DISCRETE:
            return self._process_discrete_input(e)
        else:
            return self._process_continuous_input(e)

    def reset(self):
        self.actions_queue.clear()
        self.current_action = Vec.null()

    def __call__(self, _) -> Vec:
        if self.output_type is OutputType.DISCRETE:
            if len(self.actions_queue) > 0:
                return self.actions_queue.pop(0)
            else:
                return Vec.null()
        else:
            return self.current_action.copy()

    def save_to_archive(self, archive: ZipFile, *args, **kwargs) -> bool:
        raise NotImplementedError

    def load_from_archive(self, archive: ZipFile, robot: Robot.BuildData,
                          *args, **kwargs) -> 'KeyboardController':
        raise NotImplementedError

    @staticmethod
    def inputs_types() -> List[InputType]:
        return list(InputType)

    @staticmethod
    def outputs_types() -> List[OutputType]:
        return list(OutputType)

    def _process_discrete_input(self, e: QKeyEvent):
        if e.type() == QEvent.KeyPress and e.key() in self.actions:
            self.actions_queue.append(self.actions[e.key()])
            return True
        return False

    def _process_continuous_input(self, e: QKeyEvent):
        if e.modifiers() != Qt.NoModifier:
            return False

        if e.type() == QEvent.KeyPress:
            down = True
        elif e.type() == QEvent.KeyRelease:
            down = False
        else:
            return False

        if e.key() in self.actions:
            self.down_keys[e.key()] = down
            self.current_action = sum([self.actions[k]
                                       for k, down in self.down_keys.items()
                                       if down],
                                      Vec.null())
            if (length := self.current_action.length()) > 0:
                self.current_action /= length
            return True
        return False
