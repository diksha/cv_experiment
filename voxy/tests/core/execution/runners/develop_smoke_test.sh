#!/bin/bash

set -euo pipefail

core/execution/runners/develop --scenarios_config_path data/scenario_sets/integration_test/integration_test.yaml --logging_level debug --max_concurrency 1
core/execution/runners/develop --scenarios_config_path data/scenario_sets/integration_test/integration_test.yaml --logging_level debug --max_concurrency 1 --experiment_config_path data/scenario_sets/experimental_configs/integration_test_experiment_config.yaml
