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

AWS_BIN=$(rlocation voxel/third_party/aws/aws.bash)
echo "$AWS_BIN"

JQ_BIN=$(rlocation voxel/third_party/jq/jq)

COMPONENT_ARN="arn:aws:greengrass:us-west-2:360054435465:components:$COMPONENT_NAME"
COMPONENT_RECIPE=$(realpath "$1")
ARTIFACT_FILE=$(realpath "$2")

# check to see if the version already exists
EXISTING_VERSION=$(aws greengrassv2 list-component-versions --arn "$COMPONENT_ARN" | $JQ_BIN ".componentVersions[] | select(.componentVersion == \"$COMPONENT_VERSION\")")
if [[ -n $EXISTING_VERSION ]]; then
	echo "Component version already exists"
	exit 1
fi

$AWS_BIN s3 cp "$ARTIFACT_FILE" "$COMPONENT_ARTIFACT_URI"
$AWS_BIN greengrassv2 create-component-version --inline-recipe fileb://"$COMPONENT_RECIPE"
