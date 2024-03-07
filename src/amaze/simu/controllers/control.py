import json
import logging
import pickle
from collections import namedtuple
from pathlib import Path
from typing import Union, Type, Optional
from zipfile import ZipFile

from amaze.simu import Robot
from amaze.simu.controllers.base import BaseController
from amaze.simu.controllers.keyboard import KeyboardController
from amaze.simu.controllers.random import RandomController
from amaze.simu.controllers.tabular import TabularController

logger = logging.getLogger(__name__)

CONTROLLERS: dict[str, Type[BaseController]] = {
    "random": RandomController,
    "keyboard": KeyboardController,
    "tabular": TabularController
}


def builtin_controllers():
    """ Provides the list of controllers shipped with this library """
    return CONTROLLERS.keys()


def check_types(controller: BaseController | Type[BaseController],
                robot: Robot.BuildData) -> bool:
    """ Ensure that the controller is compatible with the specified
     inputs/outputs """
    def _fmt(e_list): return ", ".join([e.name for e in e_list])
    assert robot.inputs in controller.inputs_types(), \
        (f"Input type {robot.inputs.name} is not valid for {controller}."
         f" Expected one of [{_fmt(controller.inputs_types())}]")
    assert robot.outputs in controller.outputs_types(), \
        (f"Output type {robot.outputs.name} is not valid for {controller}."
         f" Expected [{_fmt(controller.outputs_types())}]")
    return True


def controller_factory(c_type: str, c_data: dict):
    """ Create a controller of a given c_type from the given c_data """
    return CONTROLLERS[c_type.lower()](**c_data)


def save(controller: BaseController, path: Union[Path, str],
         infos: Optional[dict] = None,
         *args, **kwargs) -> Path:
    """ Save the controller under the provided path

    Optionally store the provided information for latter reference (e.g.
    type of mazes, performance, ...)
    Additional arguments are forwarded to the controller's
    :meth:`~.BaseController.save_to_archive`
    """

    reverse_map = {t: n for n, t in CONTROLLERS.items()}
    if (controller_class := reverse_map.get(type(controller), None)) is None:
        raise ValueError(f"Controller class {type(controller)} is not"
                         f" registered")

    if isinstance(path, str):
        path = Path(path)
    if path.suffix != ".zip":
        path = path.with_suffix(".zip")

    with ZipFile(path, "w") as archive:
        archive.writestr("controller_class",
                         controller_class)
        controller.save_to_archive(archive, *args, **kwargs)

        _infos = controller.infos.copy()
        _infos.update(infos)
        archive.writestr("infos",
                         json.dumps(_infos).encode("utf-8"))

    logger.debug(f"Saved controller to {path}")

    return path


def load(path: Union[Path, str], *args, **kwargs):
    """ Loads a controller from the provided path.

    Handles any type currently registered. When using extensions, make sure
    to load (import) all those used during training.
    """
    logger.debug(f"Loading controller from {path}")
    with ZipFile(path, "r") as archive:
        controller_class = archive.read("controller_class").decode("utf-8")
        logger.debug(f"> controller class: {controller_class}")
        if (c_type := CONTROLLERS.get(controller_class)) is None:
            msg = f"Unsupported controller type {controller_class}."
            if len(tokens := controller_class.split(".")) > 1:
                msg += (f" Did you forget to include the '{tokens[0]}'"
                        f" extension?")
            raise ValueError(msg)
        c = c_type.load_from_archive(archive, *args, **kwargs)
        c.infos = json.loads(archive.read("infos").decode("utf-8"))
        return c
