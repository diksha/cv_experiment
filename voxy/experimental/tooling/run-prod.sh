#!/bin/bash

while true; do
	timeout 1200 ./bazel run -- //core/execution/runners:production --camera_config_path "$1"
done
