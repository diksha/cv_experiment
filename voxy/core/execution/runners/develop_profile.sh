#!/bin/bash

set -euo pipefail

./bazel build //core/execution/runners:develop

FILEPATH=$(readlink -f "$0")
BASEDIR=$(dirname "$FILEPATH")
EXECUTION_DIR=$(dirname "$BASEDIR")
CORE_DIR=$(dirname "$EXECUTION_DIR")
WORKSPACEDIR=$(dirname "$CORE_DIR")

cd "$WORKSPACEDIR"

OUTPUT_FLAME_GRAPH_DIR_DEFAULT="$HOME/tmp/var/voxel/flamegraphs/$(uuidgen)/"
#this ignores all arguments except the output directory
index=1
all_args=()
next_arg_used=0
OUTPUT_FLAME_GRAPH_DIR="$OUTPUT_FLAME_GRAPH_DIR_DEFAULT"
for arg in "$@"; do
	case "$arg" in
	"--flame_graph_output_dir")
		next_arg_index=$((index + 1))
		OUTPUT_FLAME_GRAPH_DIR="${!next_arg_index}"
		next_arg_used=1
		;;
	*)
		if [[ $next_arg_used -eq 0 ]]; then
			all_args+=("$arg")
		fi
		next_arg_used=0
		;;
	esac
	((index += 1))
done # $@ sees arguments as separate words.
echo "Running develop graph with args: ${all_args[*]}"

echo "Output flame graph dir: $OUTPUT_FLAME_GRAPH_DIR"

bazel-out/k8-fastbuild/bin/core/execution/runners/develop "${all_args[@]}" &
pid="$!"
function cleanup() {
	if ps -p $pid >/dev/null; then
		echo "Cleaning up: $pid"
		kill -9 $pid
	fi
}
trap cleanup EXIT

# Remove previous profile.
cd "$WORKSPACEDIR"
all_args=("$@")

mkdir -p "$OUTPUT_FLAME_GRAPH_DIR"
./bazel run //third_party/py-spy:py-spy -- record -o "$OUTPUT_FLAME_GRAPH_DIR/profile.svg" --pid "$pid"

# Upload to s3
TIMESTAMP="$(date +%s)"
AWS_PATH="s3://voxel-temp/develop_graph_profiling/$USER/$TIMESTAMP.svg"
echo "Uploading file to $AWS_PATH"
./tools/aws s3 cp "$OUTPUT_FLAME_GRAPH_DIR/profile.svg" "$AWS_PATH"
echo "File can be accessed at: $AWS_PATH"
