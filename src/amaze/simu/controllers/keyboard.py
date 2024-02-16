from abc import ABCMeta
from logging import getLogger

from PyQt5.QtCore import QObject, Qt, QEvent
from PyQt5.QtGui import QKeyEvent
from PyQt5.QtWidgets import QApplication

from amaze.simu.controllers.base import BaseController
from amaze.simu.pos import Vec
from amaze.simu.robot import InputType, OutputType

logger = getLogger(__name__)


class __Meta(type(QObject), ABCMeta):
    pass


class KeyboardController(QObject, BaseController, metaclass=__Meta):
    def __init__(self, a_type: OutputType):
        super().__init__(a_type=a_type)
        QApplication.instance().installEventFilter(self)
        self.actions = {
            Qt.Key_Right: Vec(1, 0),
            Qt.Key_Up: Vec(0, 1),
            Qt.Key_Left: Vec(-1, 0),
            Qt.Key_Down: Vec(0, -1),
        }

        if self.action_type == OutputType.DISCRETE:
            self.actions_queue = []
        else:
            self.down_keys = {a: False for a in self.actions.keys()}
            self.current_action = Vec.null()

    def eventFilter(self, obj: 'QObject', e: 'QEvent') -> bool:
        if not isinstance(e, QKeyEvent):
            return False
        if self.action_type is OutputType.DISCRETE:
            return self._process_discrete_input(e)
        else:
            return self._process_continuous_input(e)

    def reset(self):
        self.current_action = Vec.null()

    def __call__(self, _) -> Vec:
        if self.action_type is OutputType.DISCRETE:
            if len(self.actions_queue) > 0:
                return self.actions_queue.pop(0)
            else:
                return Vec.null()
        else:
            return self.current_action.copy()

    def save(self):
        return {}

    def restore(self, _):
        pass

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
            if (l := self.current_action.length()) > 0:
                self.current_action /= l
            return True
        return False
