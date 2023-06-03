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

# Adapted from ./core/scripts/buildkite_smoke_test.sh

set -euo pipefail

echo "Starting ML Training Test"

METAVERSE_ENVIRONMENT="INTERNAL" ./bazel run //core/infra/sematic/perception/ml_training:main_local -- --experiment_config_path core/ml/experiments/configs/DOOR_STATE.yaml --experimenter buildkite_smoketest --camera_uuids americold/modesto/0011/cha --no-notify --silent --override_config_path core/ml/experiments/configs/test/SMOKE_TEST_DOOR_STATE.yaml
