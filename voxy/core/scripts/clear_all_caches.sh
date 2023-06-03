#!/bin/bash

set -euo pipefail

# Clear Trunk cache
./tools/trunk cache clean

# Delete all node_modules directories
rm -rf ./node_modules
rm -rf ./services/portal/web/node_modules

# Clear Bazel cache
./bazel clean --expunge_async
rm -rf ./.voxelcache
