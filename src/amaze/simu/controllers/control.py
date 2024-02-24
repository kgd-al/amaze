import json
from enum import Enum
from pathlib import Path
from typing import Union, Type

from amaze.simu.controllers.base import BaseController
from amaze.simu.controllers.keyboard import KeyboardController
from amaze.simu.controllers.random import RandomController
from amaze.simu.controllers.tabular import TabularController
from amaze.simu.types import InputType, OutputType

# from amaze.sb3.controller import SB3Controller


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


def dump(controller: BaseController, path: Union[Path, str]):
    reverse_map = {t: n for n, t in CONTROLLERS.items()}
    j = dict(type=reverse_map[type(controller)])
    j.update(controller.to_json())

    with open(path, 'w') as f:
        json.dump(j, f)


def load(path: Union[Path, str]):
    if isinstance(path, str):
        path = Path(path)
    # if path.suffix == ".zip":
    #     return SB3Controller.load(path)
    with open(path, 'r') as f:
        dct: dict = json.load(f)
        c_type = dct.pop("type")
        return CONTROLLERS[c_type.lower()].from_json(dct)
