[ -z "$VIRTUAL_ENV" ] && source ../venv/bin/activate

out=docs/_build/html
rm -r $out
rm -fr docs/src/_*
args="-Ea"

sphinx-autobuild $args docs/src/ $out \
    --ignore '*/_autogen/errors.rst' --ignore '*/_auto*' \
    --pre-build 'rm -fr docs/_build' --pre-build clear --pre-build date --watch src/amaze
