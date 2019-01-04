#!/bin/sh

[ $# -ge 2 ] || { echo Usage: $0 venv_name tensorflow_pip_specification >&2; exit 1; }
venv="$1"
tf="$2"

python3 -m venv "$venv"
"$venv"/bin/pip3 install $tf
"$venv"/bin/pip3 install cython
"$venv"/bin/pip3 install git+https://github.com/andersjo/dependency_decoding
