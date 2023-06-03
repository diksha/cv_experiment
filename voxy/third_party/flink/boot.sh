#!/bin/bash

set -euo pipefail

echo "Voxel Flink Bootstrap"

BUILD_VENV="$APP".build_venv

if [ ! -e "$BUILD_VENV" ]; then
	sed 's/RUN_BINARY_ENTRY_POINT=true/RUN_BINARY_ENTRY_POINT=false/' "$APP" >"$BUILD_VENV"
	chmod +x "$BUILD_VENV"
fi

"$BUILD_VENV"

export PATH=$FLINK_HOME/bin:/opt/voxel/bin:$PATH

# if we have credentials in /root we copy them
# to make them available to any flink process
if [[ -d /root/.aws ]]; then
	cp -r /root/.aws /opt/flink/.aws
	chown -R flink:flink /opt/flink/.aws
fi

exec /flink-docker-entrypoint.sh "$@"
