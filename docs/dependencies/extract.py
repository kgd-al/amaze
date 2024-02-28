import sys
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path

import graphviz
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

    with (open(file, "r") as f):
        for line in f:
            if line.startswith("import") or \
                    (line.startswith("from") and "import" in line):
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
    IGNORE_STDLIB = True

    g_dot = graphviz.Digraph('AMaze', format='pdf', strict=True)
    g_dot.graph_attr['rankdir'] = 'LR'
    g_dot.attr(compound="True")
    g_dot.attr(label=name)

    stdlib_modules = {True: [], False: []}

    def _process(key, _dct, parent, depth=0):
        __name = _name(key)
        with parent.subgraph(name=__name) as _cluster:
            _cluster.attr(cluster="True")
            _cluster.attr(label=key)
            for k, v in _dct.items():
                if isinstance(v, dict):
                    _process(k, v, _cluster, depth+1)
                else:
                    for d in v:
                        _cluster.node(k, label=_name(k))
                        stdlib = (d in sys.stdlib_module_names)

                        if stdlib:
                            _args = dict(shape='diamond', color='gray')
                        else:
                            _args = dict()

                        if not stdlib or not IGNORE_STDLIB:
                            stdlib_modules[stdlib].append(
                                (d, dict(label=_name(d), **_args)))
                            g_dot.edge(k, d, **_args)

    _process(name, dct, g_dot)

    for t, cluster_name in [(True, "stdlib"), (False, "dependencies")]:
        with g_dot.subgraph(name=cluster_name) as cluster:
            cluster.attr(label=cluster_name)
            cluster.attr(cluster="True")
            cluster.attr(rank="same")
            for n, args in stdlib_modules[t]:
                cluster.node(n, **args)

    return g_dot.render(directory=me.parent, filename=name, cleanup=True)


def process_dependencies_list(dicts):
    exts = "amaze.extensions"
    d_sets = {k: set() for k in dicts[exts]}
    exts_dicts = dicts.pop(exts)
    d_sets["amaze"] = set()

    def _subdicts_values(_d):
        for _k, _v in _d.items():
            if isinstance(_v, dict):
                yield from _subdicts_values(_v)
            else:
                yield from _v

    for d in _subdicts_values(dicts):
        if d not in sys.stdlib_module_names:
            d_sets["amaze"].add(d)

    for e, ed in exts_dicts.items():
        _e = e.replace(exts + ".", "")
        for d in _subdicts_values(ed):
            if d not in sys.stdlib_module_names:
                d_sets[e].add(d)

    for pkg, d_set in d_sets.items():
        if pkg == "amaze":
            continue
        d_sets[pkg] = d_set.difference(d_sets["amaze"])

    print("\nDependencies:")
    _w = max(len(_d) for d in d_sets.values() for _d in d)
    _fmt = f"{{:>{_w}s}}: {{}}"
    with open(me.parent.joinpath("dependencies"), "w") as f:
        for pkg, d_set in d_sets.items():
            pkg = pkg.replace(exts + ".", "")
            print("==", pkg, "="*(_w+5-len(pkg)))
            f.write(f"{pkg}\n")
            for d in d_set:
                try:
                    v = version(d)
                except PackageNotFoundError:
                    v = None
                print(_fmt.format(d, v))
                f.write(f"  {d}\n")


if __name__ == "__main__":
    me = Path(sys.argv[0])
    src = me.parent.parent.parent.joinpath('src').joinpath('amaze')
    assert src.exists()

    nested_dict = process_folder(src)

    process_dependencies_list(nested_dict)

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
