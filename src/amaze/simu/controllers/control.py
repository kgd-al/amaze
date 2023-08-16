import json
from enum import Enum
from pathlib import Path
from typing import Union

from amaze.simu.controllers.base import BaseController
from amaze.simu.controllers.keyboard import KeyboardController
from amaze.simu.controllers.random import RandomController
from amaze.simu.controllers.tabular import TabularController
from amaze.sb3.controller import SB3Controller


class Controllers(Enum):
    RANDOM = RandomController
    KEYBOARD = KeyboardController
    TABULAR = TabularController


def controller_factory(c_type: Controllers, c_data: dict):
    return c_type.value(**c_data)


def dump(controller: BaseController, path: Union[Path, str]):
    reverse_map = {c.value: c.name.lower() for c in Controllers}
    j = dict(type=reverse_map[type(controller)])
    j.update(controller.to_json())

    with open(path, 'w') as f:
        json.dump(j, f)


def load(path: Union[Path, str]):
    if isinstance(path, str):
        path = Path(path)
    if path.suffix == ".zip":
        return SB3Controller.load(path)
    with open(path, 'r') as f:
        dct: dict = json.load(f)
        c_type = dct.pop("type")
        c_enum = Controllers[c_type.upper()]
        return c_enum.value.from_json(dct)
