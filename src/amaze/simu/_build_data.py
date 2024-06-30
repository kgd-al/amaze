import ast
import inspect
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
        def __bool__(self):
            return False

        def __repr__(self):
            return "Unset"

    unset = Unset()

    def override_with(self, other):
        assert isinstance(other, self.__class__)
        # logger.debug(f"Overriding: {self}\n"
        #              f"      with: {other}")

        for lhs, rhs in zip(self.__fields(), other.__fields()):
            assert lhs.name == rhs.name
            if not isinstance(v := getattr(other, rhs.name), self.Unset):
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
        return "-".join(cls.__qualname__.split(".")[:-1]).lower()

    @classmethod
    def __fields(cls):
        return [field for field in fields(cls) if get_origin(field.type) is Annotated]

    @classmethod
    def _is_legal_unset(cls, allow_unset, field):
        return allow_unset and isinstance(field, cls.Unset)

    def _assert_field_type(self, field: str, allow_unset: bool, field_type=None, value_tester=None):
        field_value = getattr(self, field)
        field_type = field_type or type(getattr(self.__class__, field, None))

        if not isinstance(field_value, field_type) and not self._is_legal_unset(
            allow_unset, field_value
        ):
            try:
                casted_value = field_type(field_value)
                roundtrip_value = type(field_value)(casted_value)
                if field_value != roundtrip_value:
                    raise ValueError(
                        f"Bad roundtrip: {field_value} {type(field_value)}"
                        f" -> {casted_value} {type(casted_value)}"
                        f" -> {roundtrip_value} {type(roundtrip_value)}"
                    )
                setattr(self, field, casted_value)
            except ValueError as e:
                raise TypeError(
                    f"Wrong type for {field}: {type(field_value)}"
                    f" cannot be cast to {field_type}:\n{e}"
                )
        if value_tester is not None and isinstance(field_value, field_type):
            if not value_tester(field_value):
                func = (
                    inspect.getsource(value_tester).split("\n")[-2].replace("return ", "").lstrip()
                )
                raise ValueError(
                    f"Invalid value for {field}: x={field_value}" f" should pass {func}"
                )

    custom_classes = {}

    @classmethod
    def populate_argparser(cls, parser):
        prefix = cls.prefix()
        for field in cls.__fields():
            a_type = field.type.__args__[0]
            t_args = get_args(a_type)
            f_type, str_type = a_type, None
            default = field.default
            action = "store"

            if cls.custom_classes and (f := cls.custom_classes.get(a_type, None)):
                f_type = getattr(f, "type_parser", None) or f_type
                str_type = getattr(f, "type_name", None) or str_type
                action = getattr(f, "action", None) or action
                default = f.default

            elif get_origin(a_type) is Union and type(None) in t_args:
                f_type = t_args[0]
            elif a_type == bool:
                f_type = ast.literal_eval
                str_type = bool

            if not str_type:
                str_type = f_type

            assert str_type, f"Invalid user type {str_type} " f"(from {a_type=} {f_type=}"

            help_msg = (
                f"{'.'.join(field.type.__metadata__)}"
                f" (default: {default},"
                f" type: {str_type.__name__})"
            )
            parser.add_argument(
                f"--{prefix}-{field.name}",
                action=action,
                dest=f"{prefix}_{field.name}",
                default=None,
                metavar="V",
                type=f_type,
                help=help_msg,
            )

    @classmethod
    def from_argparse(cls, namespace, set_defaults=True):
        prefix = cls.prefix()
        data = cls()
        for field in cls.__fields():
            f_name = f"{prefix}_{field.name}"
            attr = None
            if (
                hasattr(namespace, f_name)
                and (maybe_attr := getattr(namespace, f_name)) is not None
            ):
                attr = maybe_attr
            elif not set_defaults:
                attr = cls.unset
            if attr is not None:
                setattr(data, field.name, attr)
        if post_init := getattr(data, "_post_init", None):  # pragma: no branch
            post_init(allow_unset=True)
        return data
