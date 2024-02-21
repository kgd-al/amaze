[ -z "$VIRTUAL_ENV" ] && source ../venv/bin/activate

out=docs/_build/html
sphinx-autobuild -Ea docs/src/ $out \
    --ignore '*/_autogen/errors.rst' --ignore '*/_apidoc/*' \
    --pre-build clear --pre-build date --watch src/amaze
