from dataclasses import fields, dataclass
from functools import lru_cache
from typing import get_args, get_origin, Union, Annotated
from abc import ABC


@dataclass
class BaseBuildData(ABC):
    class Unset:
        def __bool__(self): return False
        def __repr__(self): return "Unset"
    unset = Unset()

    @classmethod
    @lru_cache(maxsize=1)
    def prefix(cls):
        return "-".join(cls.__qualname__.split('.')[:-1]).lower()

    @classmethod
    def __fields(cls):
        return [field for field in fields(cls) if
                get_origin(field.type) is Annotated]

    @classmethod
    def populate_argparser(cls, parser):
        prefix = cls.prefix()
        for field in cls.__fields():
            a_type = field.type.__args__[0]
            t_args = get_args(a_type)
            if get_origin(a_type) is Union and type(None) in t_args:
                f_type = t_args[0]
            else:
                f_type = a_type

            help_msg = \
                f"{'.'.join(field.type.__metadata__)}" \
                f" (default: {field.default}," \
                f" type: {f_type})"
            parser.add_argument(f"--{prefix}-{field.name}",
                                dest=f"{prefix}_{field.name}",
                                default=None, metavar="V",
                                type=f_type, help=help_msg)

    @classmethod
    def from_argparse(cls, namespace, set_defaults=True):
        prefix = cls.prefix()
        data = cls()
        for field in cls.__fields():
            f_name = f"{prefix}_{field.name}"
            if hasattr(namespace, f_name) \
                    and (attr := getattr(namespace, f_name)) is not None:
                setattr(data, field.name, attr)
            elif set_defaults:
                setattr(data, field.name, field.default)
            else:
                setattr(data, field.name, cls.unset)
        if hasattr(data, "__post_init__"):
            data.__post_init__()
        return data
