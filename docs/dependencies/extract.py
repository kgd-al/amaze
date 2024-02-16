import pprint
import sys
from pathlib import Path

import graphviz
from graphviz import dot


def process_folder(parent, folder, depth=0):
    prefix = "\t"*depth

    folder_str = str(folder)
    with parent.subgraph(name=folder_str) as cluster:
        cluster.attr(cluster="True")
        cluster.attr(label=folder.name)
        for item in folder.glob("*"):
            if item.is_file() and item.suffix == ".py":
                print(f"[P] {prefix}{item}")
                process_file(cluster, item)
            elif item.is_dir() and item.name != "__pycache__":
                print(f"[D] {prefix}{item}")
                process_folder(cluster, item, depth+1)


def process_file(cluster, file):
    module = str(file)
    cluster.node(module, label=file.with_suffix("").name)
    with open(file, "r") as f:
        for line in f:
            if "import" in line:
                dep = line.lstrip().replace('\n', '').split(" ")[1]
                dot.edge(module, dep)
                dependencies.add(dep)


def process_dependencies():
    dct = {}

    def _process(dep_list, _dct):
        if len(dep_list) == 0:
            return

        _item = dep_list.pop(0)
        if _item not in _dct:
            _dct[_item] = {}
        _process(dep_list, _dct[_item])

    for dep in dependencies:
        _process(dep.split('.'), dct)

    pprint.pprint(dct)

    def _process(_dct, key=None, cluster=dot):
        if key is None:
            key = []
        for k, v in _dct.items():
            _key = key + [k]
            print(k, v)
            if len(v) > 0:
                with cluster.subgraph(name=k) as _cluster:
                    _cluster.attr(label=k)
                    _cluster.attr(cluster="True")
                    _process(v, _key, _cluster)
            else:
                cluster.node(name=".".join(_key), label=k)

    _process(dct)


if __name__ == "__main__":
    me = Path(sys.argv[0])
    src = me.parent.parent.parent.joinpath('src').joinpath('amaze')
    assert src.exists()

    dependencies = set()

    dot = graphviz.Digraph('AMaze', format='pdf')
    dot.graph_attr['rankdir'] = 'LR'
    dot.attr(compound="True")
    process_folder(dot, src)

    process_dependencies()
    # print(dot.source)

    # dot_file = me.parent.joinpath("data.dot")
    # pdf_file = dot_file.with_suffix(".pdf")
    dot.render(directory=me.parent, filename="amaze")
