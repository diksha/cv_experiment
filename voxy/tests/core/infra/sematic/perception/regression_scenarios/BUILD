load("@pip_deps//:requirements.bzl", "requirement")
load("//third_party/pytest:build_defs.bzl", "voxel_py_test")

voxel_py_test(
    name = "pipeline_test",
    srcs = [
        "pipeline_test.py",
    ],
    deps = [
        requirement("sematic"),
        "//core/execution/utils:perception_runner_context",
        "//core/infra/sematic/perception:run_inferences",
        "//core/infra/sematic/perception/regression_scenarios:pipeline",
        "//core/structs",
    ],
)

voxel_py_test(
    name = "add_regression_scenario_test",
    srcs = [
        "add_regression_scenario_test.py",
    ],
    deps = [
        "//core/infra/sematic/perception/regression_scenarios:add_regression_scenario",
    ],
)

voxel_py_test(
    name = "utils_test",
    srcs = [
        "utils_test.py",
    ],
    deps = [
        requirement("sematic"),
        "//core/execution/utils:perception_runner_context",
        "//core/infra/sematic/perception:performance_evaluation",
        "//core/infra/sematic/perception:run_inferences",
        "//core/infra/sematic/perception/regression_scenarios:utils",
        "//core/structs",
    ],
)
