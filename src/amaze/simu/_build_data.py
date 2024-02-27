import ast
import logging
from abc import ABC
from copy import copy
from dataclasses import fields, dataclass
from functools import lru_cache
from typing import get_args, get_origin, Union, Annotated


logger = logging.getLogger(__name__)


@dataclass
class BaseBuildData(ABC):
    class Unset:
        def __bool__(self): return False
        def __repr__(self): return "Unset"
    unset = Unset()

    def override_with(self, other):
        assert isinstance(other, self.__class__)
        # logger.debug(f"Overriding: {self}\n"
        #              f"      with: {other}")

        for lhs, rhs in zip(self.__fields(), other.__fields()):
            assert lhs.name == rhs.name
            if (v := getattr(other, rhs.name)) != self.unset:
                setattr(self, lhs.name, v)
        return self

    def where(self, **kwargs):
        other = copy(self)
        for k, v in kwargs.items():
            assert hasattr(self, k)
            setattr(other, k, v)
        return other

    @classmethod
    @lru_cache(maxsize=1)
    def prefix(cls):
        return "-".join(cls.__qualname__.split('.')[:-1]).lower()

    @classmethod
    def __fields(cls):
        return [field for field in fields(cls) if
                get_origin(field.type) is Annotated]

    custom_classes = {}

    @classmethod
    def populate_argparser(cls, parser):
        prefix = cls.prefix()
        for field in cls.__fields():
            a_type = field.type.__args__[0]
            t_args = get_args(a_type)
            f_type, str_type = a_type, None
            default = field.default
            action = 'store'

            if (cls.custom_classes and
                    (f := cls.custom_classes.get(a_type, None))):
                f_type = getattr(f, 'type_parser', None) or f_type
                str_type = getattr(f, 'type_name', None) or str_type
                if (d := getattr(f, 'default', None)) is not None:
                    default = d
                action = getattr(f, 'action', None) or action

            elif get_origin(a_type) is Union and type(None) in t_args:
                f_type = t_args[0]
            elif a_type == bool:
                f_type = ast.literal_eval
                str_type = bool

            if not str_type:
                str_type = f_type

            assert str_type, (f"Invalid user type {str_type} "
                              f"(from {a_type=} {f_type=}")

            help_msg = \
                f"{'.'.join(field.type.__metadata__)}" \
                f" (default: {default}," \
                f" type: {str_type.__name__})"
            parser.add_argument(f"--{prefix}-{field.name}",
                                action=action,
                                dest=f"{prefix}_{field.name}",
                                default=None, metavar="V",
                                type=f_type, help=help_msg)

    @classmethod
    def from_argparse(cls, namespace, set_defaults=True):
        prefix = cls.prefix()
        data = cls()
        for field in cls.__fields():
            f_name = f"{prefix}_{field.name}"
            attr = None
            if hasattr(namespace, f_name) \
                    and (maybe_attr := getattr(namespace, f_name)) is not None:
                attr = maybe_attr
            elif not set_defaults:
                attr = cls.unset
            if attr is not None:
                setattr(data, field.name, attr)
        if hasattr(data, "__post_init__"):
            data.__post_init__()
        return data
