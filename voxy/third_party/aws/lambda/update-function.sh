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

echo "FUNCTION_NAME=$FUNCTION_NAME"

AWS=$(rlocation voxel/third_party/aws/cli/aws_cli)

if [[ -n ${ZIP_FILE:-} ]]; then
	echo "Updating lambda function $FUNCTION_NAME with zip file $ZIP_FILE"

	ZIP_FILE_PATH=$ZIP_FILE

	"$AWS" lambda update-function-code --function-name "$FUNCTION_NAME" --zip-file "fileb://$ZIP_FILE_PATH"
	echo "Update started, waiting for completion"
	exec "$AWS" lambda wait function-updated --function-name "$FUNCTION_NAME"
elif [[ -n ${IMAGE_REPO:-} ]]; then
	IMAGE_TAG=$(cat "$IMAGE_TAG_FILE")
	IMAGE_URI="${IMAGE_REPO}:${IMAGE_TAG}"
	echo "Updating lambda function $FUNCTION_NAME with image $IMAGE_URI"

	"$AWS" lambda update-function-code --function-name "$FUNCTION_NAME" --image-uri "$IMAGE_URI"
	echo "Update started, waiting for completion."
	exec "$AWS" lambda wait function-updated --function-name "$FUNCTION_NAME"
else
	echo "must set ZIP_FILE or IMAGE_URI"
	exit 1
fi
