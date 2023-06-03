#!/bin/bash

# --- begin runfiles.bash initialization v2 ---
set -uo pipefail
f=bazel_tools/tools/bash/runfiles/runfiles.bash
source "${RUNFILES_DIR:-/dev/null}/$f" 2>/dev/null ||
	source "$(grep -sm1 "^$f " "${RUNFILES_MANIFEST_FILE:-/dev/null}" | cut -f2- -d' ')" 2>/dev/null ||
	source "$0.runfiles/$f" 2>/dev/null ||
	source "$(grep -sm1 "^$f " "$0.runfiles_manifest" | cut -f2- -d' ')" 2>/dev/null ||
	source "$(grep -sm1 "^$f " "$0.exe.runfiles_manifest" | cut -f2- -d' ')" 2>/dev/null ||
	{
		echo >&2 "ERROR: cannot find $f"
		exit 1
	}
f=
set -e
# --- end runfiles.bash initialization v2 ---

runfiles_export_envvars

STREAMLIT_RUNNER_REL=$(rlocation voxel/services/perception/fisheye/streamlit_runner.zip)
STREAMLIT_RUNNER=$(realpath -s "$STREAMLIT_RUNNER_REL")

FISH2PERSP_GUI_REL=$(rlocation voxel/services/perception/fisheye/fisheye_gui.py)
FISH2PERSP_GUI=$(realpath -s "$FISH2PERSP_GUI_REL")

FISH2PERSP_REL=$(rlocation voxel/third_party/fish2persp/fish2persp)
FISH2PERSP=$(realpath -s "$FISH2PERSP_REL")
export FISH2PERSP

exec python -u "$STREAMLIT_RUNNER" -- run "$FISH2PERSP_GUI" "$@"
