#!/bin/bash

set -eou pipefail

# Run a Bazel binary target from the working directory.
#
#     Usage within a tool script:
#
#         #!/bin/bash
#         source "$(dirname "$0")/_utils.sh"
#         run_bazel_binary_from_working_directory @pnpm//:pnpm -- "$@"
#
run_bazel_binary_from_working_directory() {
	FILEPATH=$(readlink -f "$0")
	BASEDIR=$(dirname "$FILEPATH")
	WORKSPACEDIR=$(dirname "$BASEDIR")
	cd "$WORKSPACEDIR" || exit 1

	BAZEL="$WORKSPACEDIR/tools/bazel"

	# Needed for certain targets, such as js_binary
	export BAZEL_BINDIR="."

	# trunk-ignore(shellcheck/SC2016): the single quotes are intentional here
	exec "$BAZEL" run --run_under='cd "$BUILD_WORKING_DIRECTORY" && exec' "$@"
}
