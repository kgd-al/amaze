import json
import logging
from enum import Enum
from pathlib import Path
from typing import Union, Type, Optional
from zipfile import ZipFile

from amaze.simu.controllers.base import BaseController
from amaze.simu.controllers.keyboard import KeyboardController
from amaze.simu.controllers.random import RandomController
from amaze.simu.controllers.tabular import TabularController
from amaze.simu.types import InputType, OutputType

logger = logging.getLogger(__name__)

CONTROLLERS = {
    "random": RandomController,
    "keyboard": KeyboardController,
    "tabular": TabularController
}


def check_types(controller: Type[BaseController],
                input_type: InputType, output_type: OutputType) -> bool:
    assert input_type == controller.inputs_type(), \
        (f"Input type {input_type} is not valid for {controller}."
         f" Expected {controller.inputs_type()}")
    assert output_type == controller.outputs_type(), \
        (f"Output type {output_type} is not valid for {controller}."
         f" Expected {controller.outputs_type()}")
    return True


def controller_factory(c_type: str, c_data: dict):
    return CONTROLLERS[c_type.lower()](**c_data)


def save(controller: BaseController, path: Union[Path, str],
         infos: Optional[dict] = None):
    reverse_map = {t: n for n, t in CONTROLLERS.items()}
    assert type(controller) in reverse_map, \
        f"Unknown controller type {type(controller)}"
    controller_class = reverse_map[type(controller)]

    if path.suffix != ".zip":
        path = path.with_suffix(".zip")

    with ZipFile(path, "w") as archive:
        archive.writestr("controller_class", controller_class)
        controller.save_to_archive(archive)

        _infos = controller.infos.copy()
        _infos.update(infos)
        archive.writestr("infos",
                         json.dumps(_infos).encode("utf-8"))

    logger.debug(f"Saved controller to {path}")

    return path


def load(path: Union[Path, str]):
    logger.debug(f"Loading controller from {path}")
    with ZipFile(path, "r") as archive:
        controller_class = archive.read("controller_class").decode("utf-8")
        logger.debug(f"> controller class: {controller_class}")
        c = CONTROLLERS[controller_class].load_from_archive(archive)
        c.infos = json.loads(archive.read("infos").decode("utf-8"))
        return c