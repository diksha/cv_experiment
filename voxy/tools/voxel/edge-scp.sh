#!/bin/bash

# --- begin runfiles.bash initialization v2 ---
# Copy-pasted from the Bazel Bash runfiles library v2.
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

export AWS_PROFILE=production-admin

AWS=$(rlocation voxel/third_party/aws/cli/aws_cli)
EDGE_SSM_NAME=$(rlocation voxel/tools/voxel/edge-ssm-name)
SESSION_MANAGER_PLUGIN=$(rlocation voxel/third_party/aws/session-manager-plugin/session-manager-plugin)
SESSION_MANAGER_PLUGIN_DIR=$(dirname "$SESSION_MANAGER_PLUGIN")
PATH=$PATH:$(realpath "$SESSION_MANAGER_PLUGIN_DIR")

exec scp -o "ProxyCommand sh -c \"$AWS ssm start-session --target \$($EDGE_SSM_NAME %h) --document-name AWS-StartSSHSession --parameters 'portNumber=%p'\"" -o "User ubuntu" "$@"
