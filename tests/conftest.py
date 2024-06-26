import itertools
import pprint
from enum import IntFlag
from typing import Dict, Any

import pytest

from _common import TestSize, generate_large_maze_data_sample, MAZE_DATA_SAMPLE

_DEBUG = 0


flags = {
    TestSize.SMALL: "--fast",
    TestSize.NORMAL: "--normal-scale",
}


def pytest_addoption(parser):
    parser.addoption(flags[TestSize.SMALL], "--small-scale",
                     action='store_const',
                     const=TestSize.SMALL, dest='scale',
                     help='Run very small test suite')
    parser.addoption(flags[TestSize.NORMAL], action='store_const',
                     const=TestSize.NORMAL, default=TestSize.NORMAL,
                     dest='scale',
                     help='Run moderate test suite')

    parser.addoption("--test-examples", dest='examples',
                     action='store_true',
                     help="Run all examples (except extensions).")

    parser.addoption("--test-extension", dest='extensions',
                     action='append', default=[],
                     help="Test requested extension.")


kgd_config: Dict[str, Dict[TestSize, Any]] = dict(
    # seed={k: range(k) for k in TestSize},
    # mutations={
    #     TestSize.SMALL: [10],
    #     TestSize.NORMAL: [100],
    #     TestSize.LARGE: [0, 1000]
    # },
    # generations={
    #     TestSize.SMALL: [25],
    #     TestSize.NORMAL: [100],
    #     TestSize.LARGE: [250]
    # },
    # ad_rate={
    #     TestSize.SMALL: [best_ad_rate],
    #     TestSize.NORMAL: [1],
    #     TestSize.LARGE: [2, .5]
    # },
    # dimension={
    #     TestSize.SMALL: [3],
    #     TestSize.NORMAL: [2, 3],
    #     TestSize.LARGE: [2, 3],
    # },
    # cppn_shape={
    #     TestSize.SMALL: [(5, 3), (3, 5)],
    #     TestSize.NORMAL: [(5, 3), (3, 5), (3, 3), (5, 5)],
    #     TestSize.LARGE: list(itertools.product([3, 4, 5], [3, 4, 5])),
    # }
)


def values_for(key: str):
    values = []
    for v in kgd_config[key].values():
        values += v
    return sorted(set(values))


def scale_for(key: str, value):
    for ts in TestSize:
        if value in kgd_config[key][ts]:
            return ts


def max_scale_for(params):
    scale = TestSize.SMALL
    for k, v in params.items():
        if k in kgd_config:
            scale = max(scale, scale_for(k, v))
    return scale


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_generate_tests(metafunc):
    def can_parametrize(name):
        if name not in metafunc.fixturenames:
            return False
        existing = [
            m for m in metafunc.definition.iter_markers('parametrize')
            if any(name == arg for arg in m.args[0].split(","))
        ]
        return len(existing) == 0

    def maybe_parametrize(name, values, short_name=None):
        if can_parametrize(name):
            # if values is None:
            #     values = values_for(name)
            # print(f"[kgd-debug] adding {name} = {values}")

            metafunc.parametrize(name, values,
                                 ids=lambda val: (
                                     "" if short_name is None
                                     else f"{short_name}_{val}"))

    # print("Configuring", metafunc.function)
    size: TestSize = metafunc.config.getoption('scale')
    maybe_parametrize("mbd_kwargs",
                      values=generate_large_maze_data_sample(size))
    maybe_parametrize("maze_str", values=MAZE_DATA_SAMPLE,
                      short_name="m")


def pytest_collection_modifyitems(config, items):
    # file_order = ["config", "genome", "innovations",
    #               "cppn", "ann", "evolution",
    #               "examples"]
    #
    # def _key(_item): return _item.reportinfo()[0].stem.split("_")[1]
    #
    # def index(_item):
    #     try:
    #         return file_order.index(_key(_item))
    #     except ValueError:
    #         return len(file_order)
    #
    # def debug_print_order(prefix):
    #     files = []
    #     for i in items:
    #         if (key := _key(i)) not in files:
    #             files.append(key)
    #     print("File order", prefix, "sort:", files)
    #
    # # debug_print_order("before")
    # items.sort(key=lambda _item: index(_item))
    # # debug_print_order("after")

    # pprint.pprint(config.option.__dict__)
    run_examples = config.getoption("examples")
    extensions = config.getoption("extensions")

    if _DEBUG > 0:
        print("*"*80)
        print("** Config:")
        print("  examples:", run_examples)
        print("extensions:", extensions)
        print("*"*80)

    def extension(path): return "extensions" in path.parts
    def testable_extension(path): return path.stem in extensions

    skip_slow = pytest.mark.skip(
        reason="Skipping slow tests with small-scale option")

    scale = config.getoption("scale")
    removed, kept = [], []
    for item in items:
        if _DEBUG > 1:
            print(item.originalname)

        keep = True

        # Specific case of extensions exemples
        if hasattr(item, 'callspec'):
            if (f := item.callspec.params.get("file")) is not None:
                if extension(f) and not testable_extension(f):
                    item.add_marker(
                        pytest.mark.skip(
                            reason=f"Only running extension example on"
                                   f" explicit request. Use --extension"
                                   f" {f.stem} to do so.")
                    )

        if extension(item.path) and not testable_extension(item.path):
            item.add_marker(
                pytest.mark.skip(
                    reason=f"Only running extension test on"
                           f" explicit request. Use --extension"
                           f" {item.path.stem} to do so.")
            )

        if item.originalname == "test_run_examples":
            if not run_examples:
                item.add_marker(
                    pytest.mark.skip(
                        reason="Only running examples test on explicit"
                               " request. Use  --test-examples to do so.")
                )

        if "slow" in item.keywords and scale < TestSize.NORMAL:
            item.add_marker(skip_slow)

        if hasattr(item, 'callspec'):
            keep = (scale >= max_scale_for(item.callspec.params))

        if keep:
            kept.append(item)
        else:
            removed.append(item)

        if _DEBUG > 1:
            print(f"> {keep=}"
                  f" skip={item.get_closest_marker('skip') is not None}")

    if removed:
        config.hook.pytest_deselected(items=removed)
        items[:] = kept
