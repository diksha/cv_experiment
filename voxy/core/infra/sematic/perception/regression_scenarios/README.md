# Regression Scenarios Pipeline

This directory contains the code for executing the regression pipeline which
executes the perception logic against many different video segments, evaluates
the results against ground-truth, and then aggregates a summary of the model
performance.

Be sure you have done the setup described in the main Sematic
[README](../../README.md).

## Execution

For executing locally, use:

```shell
bazel run //core/infra/sematic/perception/regression_scenarios:main_local -- \
    --scenario_set_path <path to something in data/scenario_sets/> \
    --inference_cluster_size <size of Ray cluster for inferences> \
    --cache_key <key for caching of perception node outputs>
```

Ex:

```shell
bazel run //core/infra/sematic/perception/regression_scenarios:main_local -- \
    --scenario_set_path data/scenario_sets/regression/piggyback.yaml \
    --inference_cluster_size 5 \
    --cache_key my-name-1
```

When you're ready to execute in the cloud, you can change the target from
`:main_local` to just `:main`. Ex:

```shell
bazel run //core/infra/sematic/perception/regression_scenarios:main -- \
    --scenario_set_path data/scenario_sets/regression/piggyback.yaml \
    --inference_cluster_size 5 \
    --cache_key my-name-1
```

This will build and push a Docker container, which takes some time
(around 15 minutes).
