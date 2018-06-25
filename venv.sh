#!/bin/sh

python3 -m venv venv
#venv/bin/pip3 install tensorflow$1==1.5.0
venv/bin/pip3 install cython
venv/bin/pip3 install git+https://github.com/andersjo/dependency_decoding
