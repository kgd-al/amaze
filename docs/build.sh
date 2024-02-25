[ -z "$VIRTUAL_ENV" ] && source ../venv/bin/activate

out=docs/_build/html
rm -r $out
rm -fr docs/src/_*
# tree -d docs

sphinx-autobuild -Ea docs/src/ $out \
    --ignore '*/_autogen/errors.rst' --ignore '*/_auto*' \
    --pre-build clear --pre-build date --watch src/amaze
