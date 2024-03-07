import importlib
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("file", Path("examples").glob("**/*.py"))
def test_run_examples(file):
    spec = importlib.util.spec_from_file_location(file.stem, file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[file.stem] = module
    spec.loader.exec_module(module)

    if hasattr(module, "main"):
        module.main()
