import copy
import logging
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

# Change cache location to /resources/cache
# change custom folder location  /resources/custom
# extension always png
# change test from path.suffix to path.exists()


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
        self.value: float = float(value)

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

        if name not in names():
            raise ValueError(f"Unknown cue name '{name}'."
                             f" Valid values are:\n"
                             f" > {', '.join(builtins())} (built-ins)\n"
                             f" > {', '.join(_custom_paths.keys())}"
                             f" (custom from {custom_resources_path()})")

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
    return Path("resources")


def cached_resources_path() -> Path:
    """ Where to look for cached built-ins """
    return resources_path().joinpath("cache")


def custom_resources_path() -> Path:
    """ Where to look for custom images """
    return resources_path().joinpath("custom")


def resources_format() -> str:
    """ Image format for the cached and custom resources """
    return "png"


# =============================================================================
# Image loader/generator

def _key_to_path(key: DataKey) -> str:
    return f"{key[0]}_{key[1]:.2}_{key[2]}.{resources_format()}"


@cache
def _factories():
    factories = {}
    for k, v in sorted(sys.modules[__name__].__dict__.items()):
        if callable(v) and k.startswith("__generator__"):
            factories[k[13:]] = v
    return factories


def _get_image(key: DataKey):
    # First look in RAM cache
    if (img := _cache.get(key)) is not None:
        return img

    # Else look in file cache
    filename = cached_resources_path().joinpath(_key_to_path(key))
    if not os.environ.get("KGD_AMAZE_NOCACHE", False):
        if filename.exists():
            img = QImage(str(filename))
            if not img.isNull():
                _cache[key] = img
                return img
            else:
                logger.warning(f"Error loading file {filename}")
                return __generator__error()

    if (factory := _factories().get(key[0])) is not None:
        img = factory(*key[1:])
        if img.isNull():
            logger.warning(f"Error generating {key}")
            return __generator__error()

        filename.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(filename))
        _cache[key] = img
        return img

    if (source := _custom_paths.get(key[0])) is not None:
        img = QImage(str(source))
        w, h = img.width(), img.height()
        if w != h and w != 4*h:
            logger.warning(f"Malformed image for {key} at {source}: Size is"
                           f" {w}x{h} which is neither a single or 4 squares")
        img = _scale_qimage(img, *key[1:])
        if img.isNull():
            logger.warning(f"Error loading {key} from {source}")
            return __generator__error()
        img.save(str(filename))
        _cache[key] = img
        return img


# =============================================================================
# Internals


def _scale_qimage(img: QImage, lightness: float, size: int):
    if lightness != _default_lightness():
        logging.warning("Specifying lightness for custom images is not yet"
                        " supported.")
    return img.scaled(size, size, transformMode=Qt.SmoothTransformation)


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
    cdict = {}
    for f in custom_resources_path().glob(f"*.{resources_format()}"):
        nlist.append(f.stem)
        cdict[f.stem] = f
    nlist.extend(builtins())
    return nlist, cdict


_names, _custom_paths = __extract_names()
_cache: dict[DataKey, QImage] = {}

# print("[kgd-debug] cached resources lists:")
# pprint.pprint(_names)
# pprint.pprint(_custom_paths)
# print("[kgd-debug] =====")


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
            subarray.append(
                copy.deepcopy(
                    _v_scale(
                        np.flipud(
                            qimage_to_numpy(img)))))
        arrays.append(subarray)

    return arrays


def qimage_to_numpy(img: QImage) -> np.ndarray:
    w, h, d = img.width(), img.height(), img.depth() // 8
    b = img.constBits().asstring(img.byteCount())
    bw = img.bytesPerLine() // d
    shape = (h, bw) if d == 1 else (h, bw, d)
    return np.ndarray(
        shape=shape, buffer=b, dtype=np.uint8)[:, :w]

