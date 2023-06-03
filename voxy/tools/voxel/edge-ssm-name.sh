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

if [ $# -ne 1 ]; then
	echo "usage: $0 <edge-uuid>"
	exit 1
fi

export AWS_PROFILE=production-admin

IOT_THING_NAME=$1

AWS=$(rlocation voxel/third_party/aws/cli/aws_cli)
JQ=$(rlocation voxel/third_party/jq/jq)

exec $AWS ssm describe-instance-information --filters "Key=ResourceType,Values=ManagedInstance" | $JQ -r ".InstanceInformationList[] | select(.SourceId==\"$IOT_THING_NAME\") | .InstanceId"
