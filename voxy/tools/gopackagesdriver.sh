#!/usr/bin/env bash

# trunk-ignore(shellcheck/SC2086)
TOOLS_PATH="$(dirname $0)"
# trunk-ignore(shellcheck/SC2086)
WORKSPACE_PATH="$(dirname $TOOLS_PATH)"

for d in bazel-bin bazel-out bazel-testlogs bazel-voxel; do
	if [[ -d $WORKSPACE_PATH/$d ]]; then
		touch "$WORKSPACE_PATH/$d"/go.mod
	fi
done

export GOPACKAGESDRIVER_BAZEL_FLAGS="--output_base=/tmp/voxel/gopackagesdriver"
exec "$WORKSPACE_PATH"/bazel run --tool_tag=gopackagesdriver -- @io_bazel_rules_go//go/tools/gopackagesdriver "${@}"
