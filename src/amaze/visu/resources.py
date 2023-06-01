import copy
import math
import sys
from functools import cache
from logging import getLogger
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image
from PyQt5.QtCore import Qt, QPoint, QRectF, QPointF
from PyQt5.QtGui import QImage, QPainter, QPainterPath, QColor, QImage, QTransform

logger = getLogger(__name__)

# =============================================================================
# Image loader/generator


def resources_path():
    return Path("data/resources/")


@cache
def _factories():
    factories = {}
    for k, v in sys.modules[__name__].__dict__.items():
        if callable(v) and k.startswith("__generator__"):
            factories[k[13:]] = v
    print(factories)
    return factories


@cache
def _get_image(name: str):
    filename = resources_path().joinpath(name + ".png")
    if filename.exists():
        img = QImage(str(filename))
        if not img.isNull():
            return img
        else:
            logger.warning(f"Error loading file {filename}")
            return __generator__error()
    else:
        tokens = name.split("_")
        img = _factories()[tokens[0]](*tokens[1:])
        if img.isNull():
            logger.warning(f"Error generating {name}")
            return __generator__error()
        else:
            filename.parent.mkdir(parents=True, exist_ok=True)
            img.save(str(filename))
            return img


# =============================================================================
# Internals

def signature(*args, **kwargs):
    """Provide explicit type conversion (for image argument parsing)"""
    def decorator(fn):
        def wrapped(*fn_args, **fn_kwargs):
            new_args = [t(raw) for t, raw in zip(args, fn_args)]
            new_kwargs = dict([(k, kwargs[k](v)) for k, v in fn_kwargs.items()])

            return fn(*new_args, **new_kwargs)

        return wrapped

    return decorator


def _image(width: int, height: Optional[int] = None):
    img = QImage(width, height if height else width, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    return img, QPainter(img)


def __generator__error():
    error, painter = _image(10)
    path = QPainterPath()
    path.moveTo(0, 0)
    path.lineTo(9, 9)
    path.moveTo(0, 9)
    path.lineTo(9, 0)
    painter.drawPath(path)
    painter.end()
    return error


@signature(int, float)
def __generator__arrow(size: int = 16, lightness: float = .5):
    arrow, painter = _image(size)
    path = QPainterPath()
    path.moveTo(0., .4)
    path.lineTo(0., .6)
    path.lineTo(.6, .6)
    path.lineTo(.6, .8)
    path.lineTo(1., .5)
    path.lineTo(.6, .2)
    path.lineTo(.6, .4)
    path.closeSubpath()
    painter.scale(size, size)
    painter.fillPath(path, QColor.fromHslF(0, 0, lightness))
    painter.end()

    return arrow


def __generator__point():
    point, painter = _image(5)
    painter.setPen(Qt.gray)
    painter.drawPoint(3, 2)
    painter.end()

    return point


@signature(int)
def __generator__hat(size):
    hat, painter = _image(4 * size, size)
    pen = painter.pen()
    pen.setColor(Qt.gray)
    pen.setWidthF(0)
    painter.setPen(pen)
    painter.scale(size, size)
    for i, a in enumerate([0, math.pi/2, math.pi, 3*math.pi/2]):
        painter.save()
        painter.translate(i+.5, .5)
        painter.drawEllipse(QRectF(-.4, -.4, .8, .8))
        painter.drawEllipse(QRectF(-.2, -.4, .4, .8))
        painter.drawLine(QPointF(0, 0),
                         .4*QPointF(math.cos(a), math.sin(a)))
        painter.restore()
    painter.end()

    return hat


# =============================================================================
# Public API


_data = {
    name: _get_image(name) for name in [
        "point",
        "arrow_15_.5", "arrow_31_.5", "arrow_15_.25",
        "hat_16", "hat_32",
        "error"
    ]
}


def image(name: str, resolution: int) -> QImage:
    return _data[name]


def qt_images(name: str, resolution: int) -> List[QImage]:
    img: QImage = _data[name]
    w, h = img.width(), img.height()
    tgt_w, tgt_h = resolution if w == h else 4*resolution, resolution
    img = img.scaled(tgt_w, tgt_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)

    img_list = []
    if w == h:
        img_list.append(img)
        for a in [90, 180, 270]:
            img_list.append(img.transformed(QTransform().rotate(a)))
    else:
        assert w == 4*h
        for i in range(4):
            img_list.append(img.copy(i*resolution, 0, resolution, resolution))
    return img_list


def np_images(name: str, resolution: int) -> List[np.ndarray]:
    def _scale(x): return x / 255.0
    _v_scale = np.vectorize(_scale)
    _images = qt_images(name, resolution)
    arrays = []
    for i, p in enumerate(_images):
        img = QImage(p.size(), QImage.Format_Grayscale8)
        img.fill(Qt.black)
        painter = QPainter(img)
        painter.drawImage(img.rect(), p, p.rect())
        painter.end()
        w, h = img.width(), img.height()
        b = img.constBits().asstring(img.byteCount())
        bw = img.bytesPerLine()
        arr = np.ndarray(shape=(h, bw),
                         buffer=b,
                         dtype=np.uint8)[:, :w]
        arrays.append(copy.deepcopy(_v_scale(arr)))

    return arrays


def resources():
    return _data.items()
