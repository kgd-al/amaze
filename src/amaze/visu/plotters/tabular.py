import logging
import math

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPainter
from PyQt5.QtWidgets import QWidget, QHBoxLayout

from amaze.simu.controllers.tabular import TabularController
from amaze.simu.types import InputType
from amaze.visu.widgets.labels import InputsLabel, ValuesLabel


logger = logging.getLogger(__name__)


def plot_inputs_values(policy: TabularController, path):
    """
    Saves a summary of the state/action pairs stored in the provided
    controller
    """
    iv_holder = QWidget()
    iv_layout = QHBoxLayout()
    iv_holder.setLayout(iv_layout)
    i_w = InputsLabel()
    iv_layout.addWidget(i_w)
    v_w = ValuesLabel()
    iv_layout.addWidget(v_w)

    wgt_h = 150
    wgt_w = 2 * wgt_h
    iv_holder.setFixedSize(wgt_w, wgt_h)
    iv_holder.setStyleSheet("background: transparent")

    margin = 10

    states = policy.states()
    n_states = len(states)
    n_w = round(math.sqrt(n_states))
    n_h = round(n_states / n_w)
    img_w = wgt_w * n_w + margin * (n_w+1)
    img_h = wgt_h * n_h + margin * (n_w+1)
    img = QImage(img_w, img_h, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    painter = QPainter(img)

    for i, state in enumerate(states):
        i_w.set_inputs(state, InputType.DISCRETE)
        v_w.set_values(policy, state)

        painter.save()
        x, y = i % n_w, i//n_w
        painter.translate(x*(wgt_w+margin)+margin,
                          y*(wgt_h+margin)+margin)
        painter.drawRect(0, 0, wgt_w, wgt_h)
        iv_holder.render(painter)
        painter.restore()

    painter.end()

    if img.save(path):
        logger.debug(f"Wrote to {path}")
