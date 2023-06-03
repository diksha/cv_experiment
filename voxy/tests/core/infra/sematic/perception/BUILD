load("@pip_deps//:requirements.bzl", "requirement")
load("//third_party/pytest:build_defs.bzl", "voxel_py_test")

voxel_py_test(
    name = "performance_evaluation_test",
    srcs = [
        "performance_evaluation_test.py",
    ],
    deps = [
        requirement("sematic"),
        "//core/infra/sematic/perception:performance_evaluation",
        "//core/structs",
    ],
)

voxel_py_test(
    name = "run_inferences_test",
    srcs = [
        "run_inferences_test.py",
    ],
    deps = [
        requirement("sematic"),
        "//core/infra/sematic/perception:graph_config",
        "//core/infra/sematic/perception:run_inferences",
        "//core/structs",
    ],
)
