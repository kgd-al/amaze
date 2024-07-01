#!/bin/bash

do_line(){
  printf "_%.0s" $(seq $(tput cols))
}

do_pip-install(){
  echo "Executing: pip install $@"
  printf "[.1] Virtual environment: $VIRTUAL_ENV\n"

  verbose="-v" #vv"
  pip install $@ $verbose
}

cmd_pretty-tree(){  # Just a regular tree but without .git folder
  where=${1:-}
  [ -z $where ] && where=$(pwd)
  tree -a -I '.git*' $where
}

cmd_clean(){  # Remove most build artifacts
  rm -rf .pytest_cache .tox
  find . -type d -a \
    \( -name build -o -name '__pycache__' -o -name "*egg-info" \) \
    | xargs rm -rf

  echo "Cleaned tree:"
  cmd_pretty-tree
}

cmd_very-clean(){  # Remove all artifacts. Reset to a clean repository
  echo "very clean"
  rm -rf tests-results/
  rm -rf tmp
  find . -name 'amaze.egg-info' | xargs rm -rf
  find src -empty -type d -delete

  cmd_clean
}

cmd_install-user(){ # Regular install (to current virtual env)
  do_pip-install .
}

cmd_install-docs(){ # Documentation install (for read the docs)
  do_pip-install .[docs]
}

cmd_install-tests(){  # Install with test in standard location
  do_pip-install .[tests]
}

cmd_install-dev(){  # Editable install (with pip)
  do_pip-install -e .[docs,tests]
#   do_manual-install 'dev-test-doc'
}

cmd_pytest(){  # Perform the test suite (small scale)
  out=tmp/tests-results
  cout=$out/coverage
  coverage=$cout/coverage.info
  coverage_txt=$cout/coverage.table
  coverage_tree=$cout/coverage.tree
  coverage_config=$cout/coverage.conf

  coverage_tmp=$(basename $coverage)
  coverage_config_tmp=$(basename $coverage_config)

  coverage_args="--rcfile=$coverage_config_tmp"
  rm $coverage_config_tmp
  touch $coverage_config_tmp

  rm -rf $out
  mkdir -p $cout

  for arg in "$@"
  do
    if [ "$arg" == "--small-scale" ]
    then
      (
        echo "[report]"
        echo "exclude_also ="
        echo "    @pytest.mark.slow"
      ) >> $coverage_config_tmp
    fi
  done

  coverage run --branch --data-file=$coverage_tmp \
    --omit "setup.py,tests/conftest.py,examples/*.py,tests/test_examples.py" \
    -m \
    pytest --durations=10 --basetemp=$out -x -ra "$@" || exit 2

  echo
  mkdir -p $cout # pytest will have cleared everything. Build it back
  mv -v $coverage_tmp $coverage
  coverage report --data-file=$coverage $coverage_args > $coverage_txt
  coverage html  --data-file=$coverage $coverage_args -d $cout/html

  echo
  echo "===="
  ls src
  echo "===="
  ls src/amaze
  echo "===="
  ls src/amaze/simu
  echo "===="
  echo "Simulation coverage:"
  set -x
  coverage report --data-file=$coverage $coverage_args \
    --skip-covered --include="src/amaze/simu/*" --fail-under=100
  set +x

  coverage_tool=~/work/utils/coverage_tree.py
  if [ -f "$coverage_tool" ]
  then
    echo
    echo "Hierarchical coverage (all):"
    ~/work/utils/coverage_tree.py $coverage --skip-covered --sort=cover $coverage_args \
    | tee $coverage_tree
  fi

  mv -v $coverage_config_tmp $coverage_config

#  line
#  coverage report --data-file=best-coverage.info
#  line

#  ((coverage report --data-file=$coverage | grep TOTAL; \
#  coverage report --data-file=best-coverage.info | grep TOTAL | sed 's/TOTAL/ BEST/') \
#  | awk '{print; for (i=2; i<=NF; i++) diff[i] = -diff[i] - $i;}END{printf " DIFF";
#          for (i=2; i<=NF; i++)
#           printf " \033[%dm%+g\033[0m", diff[i] == 0 ? 0 : 31, diff[i];
#          print"\n";}') | column -t
}

do_doc_prepare() {
  out=docs/_build
  rm -r $out
  rm -fr docs/src/_*
  mkdir -p $out
}

cmd_doc(){  # Generate the documentation
# also requires sphinx and sphinx-pyproject
  do_doc_prepare
  nitpick=-n
  sphinx-build doc/src/ $out/html -b html $nitpick -W $@ 2>&1 \
  | tee $out/log
}

cmd_autodoc(){  # Generate the documentation continuously
  do_doc_prepare

  args="-Ea"

  sphinx-autobuild $args docs/src/ $out/html \
       --watch src/amaze --watch examples --color -W --keep-going \
       -w $out/errors \
      --ignore '*/_auto*' \
      --pre-build 'rm -fr docs/_build' --pre-build clear --pre-build date \
      $@
}

cmd_before-deploy(){  # Run a lot of tests to ensure that the package is clean
  ok=1
  check(){
    if [ $ok -ne 0 ]
    then
      printf "\033[31mPackage is not ready to deploy."
      printf " See error(s) above.\033[0m\n"
    else
      printf "\033[32mPackage checks out.\033[0m\n"
    fi
  }
  trap check exit
  $(dirname $0)/deploy_tests.sh
  ok=0
}

help(){
  echo "Set of commands to help managing/installing/cleaning this repository"
  do_line
  printf "\nAvailable commands:\n"
  sed -n 's/^cmd_\(.*\)(){ *\(#\? *\)\(.*\)/\1|\3/p' $0 | column -s '|' -t
}

if [ $# -eq 0 ]
then
  echo "No commands/arguments provided"
  help
  exit 1
fi

if [ "$1" == "--venv" ]
then
  source $2/bin/activate
  shift 2

elif [ "$1" == "--no-venv" ]
then
  shift 1
  NO_VIRTUAL_ENV="yes"
fi

if [ -z $VIRTUAL_ENV ] && [ -z $NO_VIRTUAL_ENV ]
then
  echo "Refusing to work outside of a virtual environment."
  echo "Activate it beforehand or provide it with the --venv <path/to/venv> option."
  echo "If you are sure (or on a CI runner) you can also use --no-venv to make me shut up."
  exit 1
fi

if [ "$1" == "-h" ]
then
  help
  exit 0
else
  cmd="cmd_$1"
  echo "Making" $cmd
  shift
  eval $cmd "$@"
fi
