"""
Built-in basic controllers.
Library-specific extensions are provided in the corresponding extension.
"""

from .base import BaseController
from .random import RandomController
from .cheater import CheaterController
from .keyboard import KeyboardController
from .tabular import TabularController

from .control import (controller_factory, builtin_controllers)


__all__ = [
    'BaseController',
    'RandomController', 'CheaterController', 'KeyboardController',
    'TabularController',

    'controller_factory', 'builtin_controllers'
]
