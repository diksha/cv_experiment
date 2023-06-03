#!/bin/bash

# Sematic's image entrypoint expects python to be at /usr/bin/python.
# We want it to use the hermetic python from Voxel's bazel setup.
# This file works as an alias for a python interpreter which will
# point to the python included with the runfiles of a python binary
# built by bazel. It can be symlinked to /usr/bin/python to have
# Sematic use the hermetic python. Sematic also needs the working dir
# to be the one for the python binary's runfiles, so using PWD here
# should be fine.
"$PWD"/../python_x86_64-unknown-linux-gnu/bin/python3 "$@"
