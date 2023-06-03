#!/bin/bash
set -euo pipefail

pyspy_dir="$(dirname "$(readlink ../pip_deps_py_spy/*.whl)")"
pyspy_binary=$pyspy_dir/bin/py-spy
echo " [+] Starting py-spy with args: $* "
sudo env "PATH=$PATH" "$pyspy_binary" "$@"
echo " [+] Shutting down"
