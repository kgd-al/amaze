import filecmp
import itertools
import os
import re
from pathlib import Path

import numpy as np
import pytest
from PyQt5.QtGui import QImage
from numpy.testing import assert_array_equal

from amaze.misc.resources import default_builtin, default_lightness, default_size, Sign, names, image, clear_cache, \
    image_cached_path, qt_images, np_images, qimage_to_numpy, NO_FILE_CACHE, cached_resources_path, no_file_cache, \
    error_builtin, _get_image, custom_resources_path, resources_format, rebuild_signs_database


def test_globals():
    print(default_builtin())
    print(default_lightness())
    print(default_size())

    assert len(names()) > 0


def _assert_valid(sign: Sign):
    name, value = sign
    assert isinstance(name, str)
    assert isinstance(value, float)
    assert 0 < value <= 1
    assert name == sign.name
    assert name.lower() == sign.name
    assert value == sign.value


def test_sign():
    sign = Sign(default_builtin(), default_lightness())
    print(sign)
    _assert_valid(sign)

    name, value = sign
    assert name == default_builtin()
    assert value == default_lightness()

    assert Sign(value=1).name == default_builtin()
    assert Sign(name="arrow").value == default_lightness()

    assert sign == Sign(*sign)
    print(hash(sign))


@pytest.mark.parametrize("name", ["arrow", "ARROW", "ArRoW", "aRrOw"])
@pytest.mark.parametrize("value", [.1, .5, 1])
def test_signs(name, value):
    _assert_valid(Sign(name, value))


def test_error_sign():
    pytest.raises(ValueError, Sign, None, None)
    pytest.raises(TypeError, Sign, default_builtin(), None)
    pytest.raises(ValueError, Sign, default_builtin(), -10)
    pytest.raises(ValueError, Sign, default_builtin(), 0)
    pytest.raises(ValueError, Sign, default_builtin(), 10)


@pytest.mark.parametrize("s_str", [
    "arrow", "arrow-1", "1", "point"
])
def test_signs_from_string(s_str):
    sign = Sign.from_string(s_str)
    _assert_valid(sign)
    assert sign == Sign.from_string(sign.to_string())


@pytest.mark.parametrize("s_str", [
    "", "unknown_name", "arrow-foo", "foo-bar-baz",
    "arrow-0", "arrow-10", "arrow--10"
])
def test_error_signs_from_string(s_str):
    pytest.raises(ValueError, Sign.from_string, s_str)


IMAGES = ("name, value, size", [
    pytest.param(name, value, size)
    for name, value, size in itertools.product(
        names(), [.25, .5, 1.], [10, 15]
    )
])


@pytest.mark.parametrize(*IMAGES)
def test_images(name, value, size, tmp_path):
    clear_cache()

    sign = Sign(name, value)
    img = image(sign, size)
    assert not img.isNull()
    assert img.width() == size
    assert img.height() == size or img.height() == 4 * size

    path = image_cached_path(sign, size)
    assert path.exists()
    test_path = tmp_path.joinpath(path.name)
    img.save(str(test_path))
    assert test_path.exists()
    assert filecmp.cmp(path, test_path)


@pytest.mark.parametrize(*IMAGES)
def test_qt_np_images(name, value, size):
    clear_cache()

    sign = Sign(name, value)
    qt_imgs = qt_images([sign], size*2)[0]
    np_imgs = np_images([sign], size*2)[0]

    assert len(qt_imgs) == len(np_imgs)
    for qt_img, np_img in zip(qt_imgs, np_imgs):
        assert (qt_img.width(), qt_img.height()) == np_img.shape
        qt_np_img = qimage_to_numpy(qt_img.convertToFormat(QImage.Format_Grayscale8))
        np_img = np.flipud((np_img * 255).astype(np.uint8))
        assert_array_equal(qt_np_img, np_img)


def test_cache():
    os.environ[NO_FILE_CACHE] = "False"
    assert not no_file_cache()

    # ==========================================
    # Generator-based

    key = (Sign("arrow", 1), 10)

    clear_cache(verbose=True, files=True)
    assert not image_cached_path(*key).exists()

    image(*key)  # Cache miss
    assert image_cached_path(*key).exists()

    image(*key)  # Cache hit

    clear_cache(verbose=True, files=False)
    assert image_cached_path(*key).exists()
    image(*key)  # Cache miss but file hit

    clear_cache(verbose=True, files=False)
    clear_cache(verbose=True, files=True)
    assert not image_cached_path(*key).exists()

    os.environ[NO_FILE_CACHE] = "True"
    clear_cache(files=True)
    image(*key)
    assert not image_cached_path(*key).exists()
    clear_cache()

    os.environ.pop(NO_FILE_CACHE)
    clear_cache(verbose=False, files=False)


def test_error_cache(caplog):
    key = (Sign("arrow", 1), default_size())

    def _error_img(size):
        return image(Sign(error_builtin(), key[0].value), size)

    def _caplog_assert(msg):
        if any(c in msg for c in ".*"):  # regex
            assert re.search(msg, caplog.text) is not None
        else:
            assert msg in caplog.text
        caplog.clear()

    clear_cache()

    bad_path = image_cached_path(*key)
    with open(bad_path, "w") as f:
        f.write("foo")
    assert bad_path.exists()
    assert _error_img(key[1]) == image(*key)
    _caplog_assert("Error loading")

    cache = cached_resources_path()
    mode = os.stat(cache).st_mode
    clear_cache()
    try:
        cache.chmod(0o555)
        image(*key)
        _caplog_assert("Could not save")
    finally:
        cache.chmod(mode)

    pytest.raises(ValueError, image, key[0], 0)
    assert _error_img(3) == _get_image((*key[0], 0))
    _caplog_assert("Error generating")

    # ==========================================
    key = (Sign("salami", 1), default_size())
    image(*key)

    clear_cache()
    bad_custom_path = custom_resources_path().joinpath(key[0].name + f".{resources_format()}")
    bad_cache_path = image_cached_path(*key)
    print(bad_custom_path)
    try:
        print("Saving bad image")
        ok = QImage(10, 20, QImage.Format_ARGB32).save(str(bad_custom_path))
        print("Saved bad image", ok)
        assert bad_custom_path.exists()
        rebuild_signs_database()
        print("Loading it")
        image(*key)
        assert not bad_cache_path.exists()
        _caplog_assert("Malformed")

    finally:
        bad_custom_path.unlink(missing_ok=True)
        bad_cache_path.unlink(missing_ok=True)

    key = (Sign("biomorph", 1), 0)
    _get_image((*key[0], key[1]))
    _caplog_assert("Error loading .* custom")

    # assert False
