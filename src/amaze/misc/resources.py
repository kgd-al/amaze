""" Handles image generation, loading and conversion Qt <-> numpy conversion
"""

import copy
import logging
import os
import sys
from enum import Enum
from functools import cache
from logging import getLogger
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PyQt5.QtCore import Qt, QRectF, QPointF
from PyQt5.QtGui import (
    QPainter,
    QPainterPath,
    QColor,
    QImage,
    QTransform,
    QPolygonF,
)

logger = getLogger(__name__)


# =============================================================================
# Public API


_DEFAULT_BUILTIN = "arrow"
_DEFAULT_LIGHTNESS = 0.5
_DEFAULT_SIZE = 15


def default_builtin():
    """Returns the default sign shape"""
    return _DEFAULT_BUILTIN


def default_lightness():
    """Returns the default lightness of a sign"""
    return _DEFAULT_LIGHTNESS


def default_size():
    """Returns the default size at which a built-in sign will be generated"""
    return _DEFAULT_SIZE


class SignType(Enum):
    """What kind of information the related sign provides"""

    CLUE = "Clue"
    """ Helpful sign. Shows the correct direction in an intersection """

    LURE = "Lure"
    """ Unhelpful sign. Points to random direction along the path """

    TRAP = "Trap"
    """ Deceitful sign. Shows the wrong direction in an intersection """


class Sign:
    """Describe a specific kind of Sign (clue, lure, trap) with a name and
    a lightness value. Shape is derived either as a built-in function or as an
    image file"""

    sep = "-"

    CLUE = SignType.CLUE
    LURE = SignType.LURE
    TRAP = SignType.TRAP

    def __init__(
        self,
        name: str = _DEFAULT_BUILTIN,
        value: float = _DEFAULT_LIGHTNESS,
    ):
        self.name: str | Path = name
        if not isinstance(self.name, str) and not isinstance(self.name, Path):
            raise ValueError(
                f"Invalid name type: {type(self.name)}" f" should be <str> or <libpath.Path>"
            )
        self.value: float = float(value)
        if not isinstance(self.value, float):
            raise TypeError(f"Invalid value: {type(self.value)}" f" should be <float>")
        if not 0 < self.value <= 1:
            raise ValueError(f"Invalid value: {self.value}" f" should be in ]0, 1]")

    def __iter__(self):
        return iter((self.name, self.value))

    def __repr__(self):
        return f"Sign({self.name}, {self.value})"

    def __eq__(self, other):
        return tuple(self) == tuple(other)

    def __hash__(self):
        return hash(tuple(self))

    @classmethod
    def from_string(cls, s: str):
        tokens = s.split(cls.sep)
        if len(tokens) == 1:
            try:
                val = float(tokens[0])
                name = _DEFAULT_BUILTIN
            except ValueError:
                name = tokens[0]
                val = 0.5
        elif len(tokens) == 2:
            name = tokens[0]
            try:
                val = float(tokens[1])
            except ValueError:
                raise ValueError(f"'{tokens[1]} is not a valid cue value")
        else:
            raise ValueError(f"Mis-formed sign: {tokens}")

        if name not in names():
            raise ValueError(
                f"Unknown cue name '{name}'."
                f" Valid values are:\n"
                f" > {', '.join(builtins())} (built-ins)\n"
                f" > {', '.join(_custom_paths.keys())}"
                f" (custom from {custom_resources_path()})"
            )

        return cls(name, val)

    def to_string(self) -> str:
        s = []
        if self.name != _DEFAULT_BUILTIN:
            s.append(self.name)
        s.append(f"{self.value:.2g}".lstrip("0"))
        return self.sep.join(s)


Signs = List[Sign]
DataKey = Tuple[str, float, int]


def resources_path():
    """Where to look for sign images and where to cache built-ins"""
    return Path("resources")


def cached_resources_path() -> Path:
    """Where to look for cached built-ins"""
    return resources_path().joinpath("cache")


def custom_resources_path() -> Path:
    """Where to look for custom images"""
    return resources_path().joinpath("custom")


def resources_format() -> str:
    """Image format for the cached and custom resources"""
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
        if w != h and 4 * w != h:
            logger.warning(
                f"Malformed image for {key} at {source}: Size is"
                f" {w}x{h} which is neither a single or 4 squares"
            )
        img = _process_custom_image(img, *key[1:])
        if img.isNull():
            logger.warning(f"Error loading {key} from {source}")
            return __generator__error()
        logger.info(f"Using custom sign ({source}) of size {img.size()}")
        ok = img.save(str(filename))
        if ok:
            logger.info(f"Saved {key} to cache")
        else:
            logger.warning(f"Could not save {key} to cache")

        _cache[key] = img

        return img


# =============================================================================
# Internals


def _process_custom_image(img: QImage, lightness: float, size: int):
    img = img.convertToFormat(QImage.Format_ARGB32).scaledToWidth(
        size, mode=Qt.SmoothTransformation
    )

    if lightness != _DEFAULT_LIGHTNESS:
        logging.warning("Specifying lightness for custom images is not yet" " supported.")
    # depth = 4
    # for j in range(img.height()):
    #     scan = img.scanLine(j)
    #     for i in range(img.width()):
    #         rgb =

    return img


def _image(width: int, lightness: float, multi=False):
    img = QImage(width, 4 * width if multi else width, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    painter = QPainter(img)
    pen = painter.pen()
    pen.setColor(__pen_color(lightness))
    pen.setWidthF(0.1)
    painter.setPen(pen)
    painter.scale(width, width)
    return img, painter


def __pen_color(lightness: float):
    return QColor.fromHslF(0, 0, lightness)


def __generator__error(lightness: float = 0, size: int = 15):
    img, painter = _image(size, lightness)
    painter.drawLine(QPointF(0, 0), QPointF(1, 1))
    painter.drawLine(QPointF(0, 1), QPointF(1, 0))
    painter.end()

    return img


def arrow_path():
    """Returns a plain arrow"""
    path = QPainterPath()
    path.moveTo(0.0, 0.4)
    path.lineTo(0.0, 0.6)
    path.lineTo(0.6, 0.6)
    path.lineTo(0.6, 0.8)
    path.lineTo(1.0, 0.5)
    path.lineTo(0.6, 0.2)
    path.lineTo(0.6, 0.4)
    path.closeSubpath()
    return path


def __generator__arrow(lightness: float, size: int):
    img, painter = _image(size, lightness)
    painter.fillPath(arrow_path(), __pen_color(lightness))
    painter.end()

    return img


def __generator__point(lightness: float, size: int):
    img, painter = _image(size, lightness)
    painter.drawPoint(QPointF(0.75, 0.5))
    painter.end()

    return img


def __generator__warning(lightness: float, size: int):
    img, painter = _image(size, lightness)
    painter.drawPolygon(QPolygonF([QPointF(0.05, 0.05), QPointF(0.95, 0.5), QPointF(0.05, 0.95)]))
    painter.fillRect(QRectF(0.3, 0.45, 0.4, 0.1), __pen_color(lightness))
    path = QPainterPath()
    path.addEllipse(QRectF(0.15, 0.45, 0.1, 0.1))
    painter.fillPath(path, __pen_color(lightness))
    painter.end()

    return img


def __generator__forbidden(lightness: float, size: int):
    img, painter = _image(size, lightness, multi=True)
    painter.translate(0.5, 0.5)
    for i in range(4):
        painter.drawEllipse(QRectF(-0.25, -0.25, 0.5, 0.5))
        painter.rotate(-45)
        painter.drawLine(QPointF(-0.25, 0), QPointF(0.25, 0))
        painter.rotate(45 + i * 90)
        painter.drawLine(QPointF(0.25, 0), QPointF(0.4, 0))
        painter.rotate(-i * 90)
        painter.translate(0, 1)

    return img


# =============================================================================
# Public API


def builtins():
    """List of all programmatically drawn signs"""
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
cached_resources_path().mkdir(parents=True, exist_ok=True)

# print("[kgd-debug] cached resources lists:")
# pprint.pprint(_names)
# pprint.pprint(_custom_paths)
# print("[kgd-debug] =====")


def names():
    """List of all possible sign shapes, including both built-ins and those
    found in :meth:`~resources_path`"""
    return _names


def image(sign: Sign, size: int) -> QImage:
    """Return the image associated with given sign in the requested size"""
    key = (sign.name, sign.value, size)
    return _get_image(key)


def qt_images(signs: Signs, resolution: int) -> List[List[QImage]]:
    """Returns images *in Qt format* (QImage) for all provided signs, with the
    requested resolution"""
    images = []
    for sign in signs:
        img: QImage = image(sign, resolution)
        w, h = img.width(), img.height()
        tgt_w, tgt_h = resolution if w == h else resolution, 4 * resolution
        if w != tgt_w and h != tgt_h:
            img = img.scaled(tgt_w, tgt_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        img_list = []
        if w == h:
            img_list.append(img)
            for a in [90, 180, 270]:
                img_list.append(img.transformed(QTransform().rotate(a)))
        else:
            assert 4 * w == h
            for i in range(4):
                img_list.append(img.copy(0, i * resolution, resolution, resolution))
        images.append(img_list)
    return images


def np_images(signs: Signs, resolution: int, rgb_fill: int = 0) -> List[List[np.ndarray]]:
    """Returns images *in numpy format* (array) for all provided signs, with
    the requested resolution"""

    def _scale(x):
        return x / 255.0

    _v_scale = np.vectorize(_scale)
    _images = qt_images(signs, resolution)
    arrays = []
    for p_list in _images:
        subarray = []
        for p in p_list:
            img = QImage(p.size(), QImage.Format_Grayscale8)
            img.fill(QColor.fromRgb(rgb_fill))
            painter = QPainter(img)
            painter.drawImage(img.rect(), p, p.rect())
            painter.end()
            subarray.append(copy.deepcopy(_v_scale(np.flipud(qimage_to_numpy(img)))))
        arrays.append(subarray)

    return arrays


def qimage_to_numpy(img: QImage) -> np.ndarray:
    """Converts a QImage into a numpy array"""
    w, h, d = img.width(), img.height(), img.depth() // 8
    b = img.constBits().asstring(img.byteCount())
    bw = img.bytesPerLine() // d
    shape = (h, bw) if d == 1 else (h, bw, d)
    return np.ndarray(shape=shape, buffer=b, dtype=np.uint8)[:, :w]
