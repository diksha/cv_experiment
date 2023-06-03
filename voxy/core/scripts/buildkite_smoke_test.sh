#!/bin/bash
##
## Copyright 2020-2021 Voxel Labs, Inc.
## All rights reserved.
##
## This document may not be reproduced, republished, distributed, transmitted,
## displayed, broadcast or otherwise exploited in any manner without the express
## prior written permission of Voxel Labs, Inc. The receipt or possession of this
## document does not convey any rights to reproduce, disclose, or distribute its
## contents, or to manufacture, use, or sell anything that it may describe, in
## whole or in part.
##

set -euo pipefail

echo "Starting"

./bazel run //core/execution/runners:develop -- --scenarios_config_path data/scenario_sets/integration_test/integration_test.yaml --logging_level debug --max_concurrency 1 --cache_key "buildkite/smoke_test_gpu/$BUILDKITE_COMMIT"
