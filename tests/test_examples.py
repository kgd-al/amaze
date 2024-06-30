import importlib
import sys
from pathlib import Path

import psutil
import pytest

from amaze.misc.utils import qt_offscreen


@pytest.mark.parametrize(
    "file",
    [
        pytest.param(path, id=str(Path(*path.parts[1:]).with_suffix("")))
        for path in Path("examples").glob("**/*.py")
    ],
)
def test_run_examples(file, capsys):
    spec = importlib.util.spec_from_file_location(file.stem, file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[file.stem] = module
    spec.loader.exec_module(module)

    qt_offscreen(offscreen=True)

    if hasattr(module, "main"):
        with capsys.disabled():
            module.main(is_test=True)

    assert len(psutil.Process().open_files()) == 0
