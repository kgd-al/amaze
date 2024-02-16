from abc import ABCMeta
from logging import getLogger

from PyQt5.QtCore import QObject, Qt, QEvent
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
            self.current_action = Vec.null()

    def eventFilter(self, obj: 'QObject', e: 'QEvent') -> bool:
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
            r = self.current_action / l \
                if (l := self.current_action.length()) > 0 \
                else self.current_action

            return r.copy()

    def save(self):
        return {}

    def restore(self, _):
        pass

    def _process_discrete_input(self, e):
        if e.type() == QEvent.KeyPress and e.key() in self.actions:
            self.actions_queue.append(self.actions[e.key()])
            return True
        return False

    def _process_continuous_input(self, e):
        if e.type() == QEvent.KeyPress:
            sign = +1
        elif e.type() == QEvent.KeyRelease:
            sign = -1
        else:
            return False

        if e.key() in self.actions:
            self.current_action += sign * self.actions[e.key()]
            return True
        return False
