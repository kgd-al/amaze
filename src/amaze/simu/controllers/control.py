import json
import logging
from pathlib import Path
from typing import Union, Type, Optional
from zipfile import ZipFile

from . import (
    BaseController,
    CheaterController,
    KeyboardController,
    RandomController,
    TabularController,
)
from .base import Robot

logger = logging.getLogger(__name__)

CONTROLLERS: dict[str, Type[BaseController]] = {
    t.short_name: t
    for t in [
        RandomController,
        CheaterController,
        KeyboardController,
        TabularController,
    ]
}


def builtin_controllers():
    """Provides the list of controllers shipped with this library"""
    return list(CONTROLLERS.keys())


def controller_factory(c_type: str, c_data: dict):
    """Create a controller of a given c_type from the given c_data"""
    c_class = CONTROLLERS[c_type.lower()]
    if not getattr(c_class, "cheats", False):
        c_data.pop("simulation", None)
    return c_class(**c_data)


def save(
    controller: BaseController,
    path: Union[Path, str],
    infos: Optional[dict] = None,
    *args,
    **kwargs,
) -> Path:
    """Save the controller under the provided path

    Optionally store the provided information for latter reference (e.g.
    type of mazes, performance, ...)
    Additional arguments are forwarded to the controller's
    :meth:`~.BaseController._save_to_archive`
    """

    reverse_map = {t: n for n, t in CONTROLLERS.items()}
    if (controller_class := reverse_map.get(type(controller), None)) is None:
        raise ValueError(f"Controller class {type(controller)} is not" f" registered")

    if isinstance(path, str):
        path = Path(path)
    if path.suffix != ".zip":
        path = path.with_suffix(".zip")

    with ZipFile(path, "w") as archive:
        archive.writestr("controller_class", controller_class)
        archive.writestr("robot", controller.robot_data.to_string())

        _infos = controller.infos.copy()
        if infos is not None:
            _infos.update(infos)
        if _infos:
            archive.writestr("infos", json.dumps(_infos).encode("utf-8"))

        # noinspection PyProtectedMember
        controller._save_to_archive(archive, *args, **kwargs)

    logger.debug(f"Saved controller to {path}")

    return path


def load(path: Union[Path, str], *args, **kwargs):
    """Loads a controller from the provided path.

    Handles any type currently registered. When using extensions, make sure
    to load (import) all those used during training.
    """
    logger.debug(f"Loading controller from {path}")
    with ZipFile(path, "r") as archive:
        controller_class = archive.read("controller_class").decode("utf-8")

        if (c_type := CONTROLLERS.get(controller_class)) is None:
            msg = f"Unsupported controller type {controller_class}."
            if len(tokens := controller_class.split(".")) > 1:
                msg += f" Did you forget to include the '{tokens[0]}'" f" extension?"
            raise ValueError(msg)

        logger.debug(f"> controller class: {controller_class}")

        robot = Robot.BuildData.from_string(archive.read("robot").decode("utf-8"))
        logger.debug(f"> Robot build data: {robot}")

        # noinspection PyProtectedMember
        c = c_type._load_from_archive(archive, *args, robot=robot, **kwargs)
        if "infos" in archive.namelist():
            c.infos = json.loads(archive.read("infos").decode("utf-8"))
        return c
