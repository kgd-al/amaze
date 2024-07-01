import pytest

from amaze.misc.resources import default_builtin, default_lightness, default_size, Sign


def test_globals():
    print(default_builtin())
    print(default_lightness())
    print(default_size())


def test_sign():
    print("Hllo")


def test_error_sign():
    pytest.raises(ValueError, Sign, None, None)
