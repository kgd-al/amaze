import itertools
import pprint
from enum import IntFlag
from typing import Dict, Any

import pytest


_DEBUG = 0


class TestSize(IntFlag):
    SMALL = 2
    NORMAL = 8
    LARGE = 16


flags = {
    TestSize.SMALL: "--fast",
    TestSize.NORMAL: "--normal-scale",
    TestSize.LARGE: "--full-scale",
}


def pytest_addoption(parser):
    parser.addoption(flags[TestSize.SMALL], "--small-scale",
                     action='store_const',
                     const=TestSize.SMALL, dest='size',
                     help='Run very small test suite')
    parser.addoption(flags[TestSize.NORMAL], action='store_const',
                     const=TestSize.NORMAL, default=TestSize.NORMAL,
                     dest='size',
                     help='Run moderate test suite')
    parser.addoption(flags[TestSize.LARGE], "--large-scale",
                     action='store_const',
                     const=TestSize.LARGE, dest='size',
                     help='Run large test suite. '
                          'Warning: While it ensures good coverage, it will'
                          ' take (too) long')

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


def pytest_generate_tests(metafunc):
    def can_parametrize(name):
        if name not in metafunc.fixturenames:
            return False
        existing = [
            m for m in metafunc.definition.iter_markers('parametrize')
            if name == m.args[0]
        ]
        return len(existing) == 0

    def maybe_parametrize(name, short_name=None, values=None):
        if can_parametrize(name):
            # print(f"adding {name} = {values_for(name)}")
            if values is None:
                values = values_for(name)

            metafunc.parametrize(name, values,
                                 ids=lambda val: ""
                                 if short_name is None else
                                 f"{short_name}_{val}")

    # # print("Configuring", metafunc.function)
    # maybe_parametrize("mutations", "m")
    # maybe_parametrize("generations", "g")
    # maybe_parametrize("seed", "s")
    # maybe_parametrize("ad_rate", "ar")
    # maybe_parametrize("dimension", "d")
    # maybe_parametrize("cppn_type", "cg", [CPPN, CPPN2D, CPPN3D])
    # maybe_parametrize("cppn_shape", "cs")
    # maybe_parametrize("cppn_nd_type", "cn", [CPPN2D, CPPN3D])
    # maybe_parametrize("eshn_genome", "eg", [True, False])
    # maybe_parametrize("verbose", "v",
    #                   [metafunc.config.getoption("verbose")])
    #
    # if can_parametrize("evo_config"):
    #     evo_config = metafunc.config.getoption('evolution')
    #     size = metafunc.config.getoption('size')
    #     sizes = {TestSize.SMALL: 1, TestSize.NORMAL: 2, TestSize.LARGE: 4}
    #     evo_config_values = [None for _ in range(sizes[size])]
    #     if evo_config is not None:
    #         for i in range(sizes[size]):
    #             dct = dict()
    #
    #             def maybe_assign(key, if_true, if_false):
    #                 dct[key] = (if_true(evo_config[key]) if
    #                             evo_config[key] is not None else if_false)
    #             maybe_assign("seed", lambda s: s+1, i)
    #             maybe_assign("population", lambda s: s, 8*size)
    #             maybe_assign("generations", lambda s: s, 8*size)
    #             dct["fitness"] = evo_config["fitness"]
    #             evo_config_values[i] = dct
    #
    #     maybe_parametrize("evo_config", short_name=None,
    #                       values=evo_config_values)


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

    scale = config.getoption("size")
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
