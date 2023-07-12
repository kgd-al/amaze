from abc import ABCMeta

from PyQt5.QtCore import QObject, Qt, QEvent
from PyQt5.QtWidgets import QApplication

from amaze.simu.controllers.base import BaseController
from amaze.simu.pos import Vec


class __Meta(type(QObject), ABCMeta):
    pass


class KeyboardController(QObject, BaseController, metaclass=__Meta):
    def __init__(self):
        super().__init__()
        QApplication.instance().installEventFilter(self)
        self.actions = {
            Qt.Key_Right: Vec(1, 0),
            Qt.Key_Up: Vec(0, 1),
            Qt.Key_Left: Vec(-1, 0),
            Qt.Key_Down: Vec(0, -1),
        }
        self.current_action = Vec.null()

    def eventFilter(self, obj: 'QObject', e: 'QEvent') -> bool:
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

    def reset(self):
        self.current_action = Vec.null()

    def __call__(self, *_) -> Vec:
        # if self.current_action is None:
        #     return Vec.
        # else:
        #     a = self.current_action
        #     self.current_action = None
        #     return a
        return self.current_action / l \
                if (l := self.current_action.length()) > 0 \
                else self.current_action

    def save(self):
        return {}

    def restore(self, _):
        pass
