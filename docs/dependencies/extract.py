import pprint
import sys
from pathlib import Path

import graphviz
from graphviz import dot

from PyPDF2 import PdfMerger


def _to_package_name(path):
    return ("amaze."
            + (str(path)
               .replace(str(src) + "/", "")
               .replace(".py", "")
               .replace("/", ".")))


def _package(module):
    return ".".join(module.split(".")[:-1])


def _name(module):
    return module.split(".")[-1]


def process_folder(folder, depth=0):
    prefix = "\t"*depth

    dct = {}

    for item in folder.glob("*"):
        k = _to_package_name(item)
        if item.is_file() and item.suffix == ".py":
            print(f"[P] {prefix}{item}")
            if items := process_file(item):
                dct[k] = items
        elif item.is_dir() and item.name != "__pycache__":
            print(f"[D] {prefix}{item}")
            dct[k] = process_folder(item, depth+1)

    return dct


def process_file(file):
    module = _to_package_name(file)

    tokens = module.split(".")
    if tokens[-1] == "__init__":
        return

    items = []

    with open(file, "r") as f:
        for line in f:
            if "import" in line:
                if "amaze" in line:
                    continue
                dep = (line.lstrip()
                       .replace('\n', '')
                       .split(" ")[1])
                if dep.startswith("amaze"):
                    # print(dep)
                    dep = ".".join(dep.split(".")[1:-1])
                    # print(">", dep)
                else:
                    dep = dep.split(".")[0]
                items.append(dep)

    return items


def _subdicts(dct):
    for k, v in dct.items():
        if isinstance(v, dict):
            yield k, v
            yield from _subdicts(v)


def process_nested_graphs(name, dct):

    dot = graphviz.Digraph('AMaze', format='pdf', strict=True)
    dot.graph_attr['rankdir'] = 'LR'
    dot.attr(compound="True")

    def _process(key, _dct, parent, depth=0):
        __name = _name(key)
        with parent.subgraph(name=__name) as cluster:
            cluster.attr(cluster="True")
            cluster.attr(label=key)
            for k, v in _dct.items():
                if isinstance(v, dict):
                    _process(k, v, cluster, depth+1)
                else:
                    for d in v:
                        cluster.node(k, label=_name(k))
                        dot.edge(k, d)

    _process(name, dct, dot)

    return dot.render(directory=me.parent, filename=name, cleanup=True)


if __name__ == "__main__":
    me = Path(sys.argv[0])
    src = me.parent.parent.parent.joinpath('src').joinpath('amaze')
    assert src.exists()

    nested_dict = process_folder(src)

    files = [
        process_nested_graphs(name, subgraph)
        for name, subgraph in _subdicts(nested_dict)
    ]

    merger = PdfMerger(strict=False)
    for file in files:
        with open(file, 'rb') as f:
            merger.append(f)
        Path(file).unlink()

    merger.write(me.parent.joinpath("amaze.pdf"))
    merger.close()
