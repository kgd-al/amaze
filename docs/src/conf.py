# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import importlib
import os
import sys
import warnings
from pathlib import Path
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
from typing import List

from sphinx.directives.code import LiteralInclude
from sphinx.util import logging
from sphinx_pyproject import SphinxConfig

# -- Ensure up-to-date sources -----------------------------------------------
for module in list(m for m in sys.modules.values() if "amaze" in m.__name__):
    print(f"Reloading {module.__name__}\r", end='')
    importlib.reload(module)
print("[kgd-debug] All amaze modules reloaded.")

# -- Project information -----------------------------------------------------

config = SphinxConfig("../../pyproject.toml")

# -- Project information -----------------------------------------------------

project = 'amaze'
copyright = '2024, ' + config.author
author = config.author

# The full version, including alpha/beta/rc tags
release = config.version

# -- Configuration for sphinx_qt_documentation --------------------------------

warnings.filterwarnings('ignore',
                        message="nodes.Text:",
                        category=DeprecationWarning)

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.autosummary',
    'sphinx_copybutton',
    'sphinx_qt_documentation',
    'myst_parser'
]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []
python_maximum_signature_line_length = 80

# -- Options for HTML sections ------------------------------------------------
autosectionlabel_prefix_document = True  # Make sure the target is unique

# -- Options for HTML output --------------------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/dev', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'stable-baselines3':
        ('https://stable-baselines3.readthedocs.io/en/master/', None),
    'gymnasium': ('https://gymnasium.farama.org/', None),
    'pytorch': ('https://pytorch.org/docs/stable/', None)
}

# -- Options for autodoc ------------------------------------------------------
autodoc_default_options = {
    "no-imported-members": True,
    "members": True,
    "no-undoc-members": True,
    "no-private-members": True,
    "member-order": "bysource",
    "ignore-module-all": True
}
# autodoc_typehints = 'description'
# autodoc_typehints_description_target = 'documented_params'

# -- Options for HTML output --------------------------------------------------

# General API configuration
# object_description_options = [
#     ("py:.*", dict(include_rubrics_in_toc=True)),
# ]

# -- Options for HTML output -------------------------------------------------

html_theme = 'furo'

html_theme_options = {
    # 'page_width': 'auto',
    'nosidebar': False,
    'body_max_width': '100%',
    'enable_search_shortcuts': True,
    'show_relbars': True,

    # 'font_family': 'monospace',
    "light_logo": "favicon-light.png",
    "dark_logo": "favicon-dark.png",
    "sidebar_hide_name": True,

    'github_banner': True,
    'github_button': True,
    'github_repo': project,
    'github_user': 'kgd-al',
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['static']

html_favicon = "static/favicon-dark.png"

html_css_files = ['custom.css']

# -- Infuriating checker for undocumented items ------------------------------

logger = logging.getLogger(__name__)
kgd_logger: str = None
kgd_verbose = False

# set up the types of member to check that are documented
members_to_watch = ['function', ]

# autodoc_default_options['undoc-members'] = True

keep_warnings = True
nitpicky = True

todo_include_todos = True


def kgd_init_log():
    wd = os.path.dirname(__file__)
    if wd is not None:
        global kgd_logger
        wd += "/_autogen"
        os.makedirs(wd, exist_ok=True)
        kgd_logger = f"{wd}/errors.rst"
        print("[kgd] Logging custom errors to", kgd_logger)
        kgd_log(header=True)


def kgd_log(message="", header=False):
    if not header:
        logger.warning(message, type="kgd", subtype="pedantic")
    with open(kgd_logger, mode="w" if header else "a") as log:
        if not header:
            message = f".. warning:: {message}"
        log.write(message + "\n\n")


def contains(pattern: str, docstring: List[str]):
    return any(pattern in ln for ln in docstring)


def simplify(full_namespace, lines, multiline, replacement=""):
    if contains(full_namespace, lines):
        for i in range(len(lines)):
            lines[i] = lines[i].replace(full_namespace, replacement)
        append(lines, '.. note:: Simplified namespace', multiline)


def append(who, what, multiline):
    if multiline:
        who.append(what)
    # else:
    #     who[-1] += what


def process(what, name, obj, lines, multiline):
    # if len(lines) == 0:
    #     kgd_log(f"Undocumented {what} {name} = {obj}({type(obj)})")
    #     append(lines, f'.. warning:: Undocumented: {what} {name}', multiline)

    if kgd_verbose:
        print(f"[kgd] processing({what}, {name}, {obj}, {lines}, {multiline}")

    if name.endswith("__init__") and len(lines) == 2 and lines[1] == "None":
        del lines[1]


def autodoc_process_docstring(app, what, name, obj, options, lines):
    if kgd_verbose:
        print(f"[kgd] autodoc-process-docstring("
              f"{what}, {name}, {obj}, {lines})")

    process(what, name, obj, lines, True)


def autodoc_skip_member(app, what, name, obj, skip, options):
    # print("[kgd] autodoc_skip_member")
    if kgd_verbose:
        print(f"[kgd] autodoc-skip-member("
              f"{what}, {name}, {obj}, {skip})")

    if obj.__doc__ and "skip-internal" in obj.__doc__:
        skip = True

    return skip


def process_signature(app, what, name, obj, options, signature, return_ant):
    # print("[kgd] autodoc_process_signature")
    if kgd_verbose:
        print(f"[kgd] autodoc-process-signature({what}, {name}, {obj},"
              f" {signature}, {return_ant})")
    lines = [i for i in [signature, return_ant] if i is not None]
    process(what, name, obj, lines, False)
    r = lines + [None for _ in range(2 - len(lines))]

    return r


# Inspired by https://stackoverflow.com/a/53267394
class DynamicLiteralInclude(LiteralInclude):
    required_arguments = 0
    optional_arguments = 1

    def run(self) -> List:
        doc = self.state.document
        root = Path(self.state.document.settings.env.app.confdir).parent.parent
        wd = Path(doc.current_source).parent
        reverse_wd = Path("/".join([".." for _
                                    in wd.relative_to(root).parents]))

        source = None
        substitutions = {}
        for sbs in doc.substitution_defs.values():
            _, key, _, value = sbs.rawsource.split()
            absolute_path = root.joinpath(value)
            if absolute_path.exists():
                source = (value, absolute_path, reverse_wd.joinpath(value))

            else:
                substitutions[key] = value

        assert source[1].exists(), \
            (f"Could not find source file at {source[0]}."
             f"Did you include .. |FILE| replace:: <path-to-file>?")

        if len(self.arguments) > 0:
            self.options["lines"] = self.arguments[0]
        elif "pyobject" in self.options:
            obj = str(source[0]).replace(".py", ":")
            obj += self.options["pyobject"]
            self.options["caption"] = f"Full listing for {obj}"
        else:
            self.options["caption"] = f"Full listing for {source[0]}"

        self.options["lineno-match"] = True
        self.arguments = [str(source[2])]

        return super().run()


def setup(app):
    kgd_init_log()
    app.connect('autodoc-process-docstring', autodoc_process_docstring)
    app.connect('autodoc-process-signature', process_signature)
    app.add_directive('kgd-literal-include', DynamicLiteralInclude)
