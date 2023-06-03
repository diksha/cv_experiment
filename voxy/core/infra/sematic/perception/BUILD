#
# Copyright 2020-2021 Voxel Labs, Inc.
# All rights reserved.
#
# This document may not be reproduced, republished, distributed, transmitted,
# displayed, broadcast or otherwise exploited in any manner without the express
# prior written permission of Voxel Labs, Inc. The receipt or possession of this
# document does not convey any rights to reproduce, disclose, or distribute its
# contents, or to manufacture, use, or sell anything that it may describe, in
# whole or in part.
#

load("@pip_deps//:requirements.bzl", "requirement")
load("@rules_python//python:defs.bzl", "py_library")

py_library(
    name = "graph_config",
    srcs = ["graph_config.py"],
    data = [
        "//configs",
    ],
    visibility = ["//visibility:public"],
    deps = [],
)

py_library(
    name = "performance_evaluation",
    srcs = ["performance_evaluation.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//core/execution/utils:graph_config_utils",
        "//core/infra/sematic/shared:resources",
        "//core/scenarios:evaluate_performance",
        "//core/structs",
        requirement("sematic"),
    ],
)

py_library(
    name = "run_inferences",
    srcs = ["run_inferences.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":graph_config",
        "//core/execution/graphs:develop",
        "//core/execution/utils:graph_config_utils",
        "//core/infra/sematic/shared:resources",
        "//core/structs",
        requirement("ray"),
        requirement("sematic"),
        requirement("loguru"),
        "//core/execution/utils:perception_runner_context",
    ],
)
