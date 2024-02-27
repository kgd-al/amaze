import copy
import os
import pprint
import sys
from enum import Enum
from functools import cache
from logging import getLogger
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PyQt5.QtCore import Qt, QRectF, QPointF
from PyQt5.QtGui import QPainter, QPainterPath, QColor, QImage, QTransform, QPolygonF

logger = getLogger(__name__)


# =============================================================================
# Public API

def _default_builtin() -> str: return "arrow"
def _default_lightness() -> float: return .5
def _default_size() -> int: return 15


class SignType(Enum):
    """ What kind of information the related sign provides """

    CLUE = "Clue"
    """ Helpful sign. Shows the correct direction in an intersection """

    LURE = "Lure"
    """ Unhelpful sign. Points to random direction along the path """

    TRAP = "Trap"
    """ Deceitful sign. Shows the wrong direction in an intersection """


class Sign:
    """ Describe a specific kind of Sign (clue, lure, trap) with a name and
    a lightness value. Shape is derived either as a built-in function or as an
    image file"""
    sep = '-'

    CLUE = SignType.CLUE
    LURE = SignType.LURE
    TRAP = SignType.TRAP

    def __init__(self, name: str = _default_builtin(),
                 value: float = _default_lightness()):
        self.name: str | Path = name
        self.value: float = value

    def __iter__(self): return iter((self.name, self.value))

    def __repr__(self): return f"Sign({self.name}, {self.value})"

    def __eq__(self, other): return tuple(self) == tuple(other)

    @classmethod
    def from_string(cls, s: str):
        tokens = s.split(cls.sep)
        if len(tokens) == 1:
            try:
                val = float(tokens[0])
                name = _default_builtin()
            except ValueError:
                name = tokens[0]
                val = .5
        elif len(tokens) == 2:
            name = tokens[0]
            try:
                val = float(tokens[1])
            except ValueError:
                raise ValueError(
                    f"'{tokens[1]} is not a valid cue value")
        else:
            raise ValueError(f"Mis-formed sign: {tokens}")

        p_name = Path(name)
        if p_name.suffix:
            if p_name.exists():
                name = p_name
            else:
                raise ValueError(f"Path like argument '{p_name}' does not"
                                 f" refer to an existing file")

        elif name not in builtins():
            raise ValueError(f"Unknown cue name '{name}'."
                             f" Valid values are {pprint.pformat(builtins())}")

        return cls(name, val)

    def to_string(self) -> str:
        s = []
        if self.name != _default_builtin():
            s.append(self.name)
        s.append(f"{self.value:.2g}".lstrip('0'))
        return self.sep.join(s)


Signs = List[Sign]
DataKey = Tuple[str, float, int]


def resources_path():
    """ Where to look for sign images and where to cache built-ins """
    return Path("cache/resources/")


# =============================================================================
# Image loader/generator

def _key_to_path(key: DataKey) -> str:
    return f"{key[0]}_{key[1]:.2}_{key[2]}.png"


@cache
def _factories():
    factories = {}
    for k, v in sorted(sys.modules[__name__].__dict__.items()):
        if callable(v) and k.startswith("__generator__"):
            factories[k[13:]] = v
    return factories


@cache
def _get_image(key: DataKey):
    filename = resources_path().joinpath(_key_to_path(key))

    if False and not os.environ["KGD_AMAZE_NOCACHE"]:
        if filename.exists():
            img = QImage(str(filename))
            if not img.isNull():
                return img
            else:
                logger.warning(f"Error loading file {filename}")

    img = _factories()[key[0]](*key[1:])
    if not img.isNull():
        filename.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(filename))
        return img
    else:
        logger.warning(f"Error generating {key}")
        return __generator__error()


# =============================================================================
# Internals


def _image(width: int, lightness: float, multi=False):
    img = QImage(width, 4 * width if multi else width, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    painter = QPainter(img)
    pen = painter.pen()
    pen.setColor(__pen_color(lightness))
    pen.setWidthF(.1)
    painter.setPen(pen)
    painter.scale(width, width)
    return img, painter


def __pen_color(lightness: float): return QColor.fromHslF(0, 0, lightness)


def __generator__error(lightness: float = 0, size: int = 15):
    img, painter = _image(size, lightness)
    painter.drawLine(QPointF(0, 0), QPointF(1, 1))
    painter.drawLine(QPointF(0, 1), QPointF(1, 0))
    painter.end()

    return img


def _arrow_path():
    path = QPainterPath()
    path.moveTo(0., .4)
    path.lineTo(0., .6)
    path.lineTo(.6, .6)
    path.lineTo(.6, .8)
    path.lineTo(1., .5)
    path.lineTo(.6, .2)
    path.lineTo(.6, .4)
    path.closeSubpath()
    return path


def __generator__arrow(lightness: float, size: int):
    img, painter = _image(size, lightness)
    painter.fillPath(_arrow_path(), __pen_color(lightness))
    painter.end()

    return img


def __generator__point(lightness: float, size: int):
    img, painter = _image(size, lightness)
    painter.drawPoint(QPointF(.75, .5))
    painter.end()

    return img


def __generator__warning(lightness: float, size: int):
    img, painter = _image(size, lightness)
    painter.drawPolygon(
        QPolygonF([QPointF(.05, .05), QPointF(.95, .5), QPointF(.05, .95)]))
    painter.fillRect(QRectF(.3, .45, .4, .1), __pen_color(lightness))
    path = QPainterPath()
    path.addEllipse(QRectF(.15, .45, .1, .1))
    painter.fillPath(path, __pen_color(lightness))
    painter.end()

    return img


def __generator__forbidden(lightness: float, size: int):
    img, painter = _image(size, lightness, multi=True)
    painter.translate(.5, .5)
    for i in range(4):
        painter.drawEllipse(QRectF(-.25, -.25, .5, .5))
        painter.rotate(-45)
        painter.drawLine(QPointF(-.25, 0), QPointF(.25, 0))
        painter.rotate(45+i*90)
        painter.drawLine(QPointF(.25, 0), QPointF(.4, 0))
        painter.rotate(-i*90)
        painter.translate(0, 1)

    return img


# =============================================================================
# Public API

def builtins():
    """ List of all programmatically drawn signs """
    return list(_factories().keys())


def __extract_names():
    nlist = []
    for f in resources_path().glob("*.png"):
        if f.name.count('_') == 0:  # Base (user-provided) file
            nlist.append(f.name)
    nlist.extend(builtins())
    return nlist


_names = __extract_names()


def names():
    """ List of all possible sign shapes, including both built-ins and those
    found in :meth:`~resources_path`"""
    return _names


def image(sign: Sign, size: int) -> QImage:
    """ Return the image associated with given sign in the requested size """
    key = (sign.name, sign.value, size)
    return _get_image(key)


def qt_images(signs: Signs, resolution: int) -> List[List[QImage]]:
    """ Returns images *in Qt format* (QImage) for all provided signs, with the
    requested resolution"""
    images = []
    for sign in signs:
        img: QImage = image(sign, resolution)
        w, h = img.width(), img.height()
        tgt_w, tgt_h = resolution if w == h else resolution, 4*resolution
        if w != tgt_w and h != tgt_h:
            img = img.scaled(tgt_w, tgt_h,
                             Qt.KeepAspectRatio, Qt.SmoothTransformation)

        img_list = []
        if w == h:
            img_list.append(img)
            for a in [90, 180, 270]:
                img_list.append(img.transformed(QTransform().rotate(a)))
        else:
            assert 4*w == h
            for i in range(4):
                img_list.append(
                    img.copy(0, i*resolution, resolution, resolution))
        images.append(img_list)
    return images


def np_images(signs: Signs, resolution: int,
              rgb_fill: int = 0) -> List[List[np.ndarray]]:
    """ Returns images *in numpy format* (array) for all provided signs, with
    the requested resolution"""
    def _scale(x): return x / 255.0
    _v_scale = np.vectorize(_scale)
    _images = qt_images(signs, resolution)
    arrays = []
    for i, p_list in enumerate(_images):
        subarray = []
        for j, p in enumerate(p_list):
            img = QImage(p.size(), QImage.Format_Grayscale8)
            img.fill(QColor.fromRgb(rgb_fill))
            painter = QPainter(img)
            painter.drawImage(img.rect(), p, p.rect())
            painter.end()
            w, h = img.width(), img.height()
            b = img.constBits().asstring(img.byteCount())
            bw = img.bytesPerLine()
            arr = np.ndarray(shape=(h, bw),
                             buffer=b,
                             dtype=np.uint8)[:, :w]
            subarray.append(copy.deepcopy(_v_scale(np.flipud(arr))))
        arrays.append(subarray)

    return arrays
