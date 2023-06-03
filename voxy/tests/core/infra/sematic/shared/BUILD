load("@pip_deps//:requirements.bzl", "requirement")
load("//third_party/pytest:build_defs.bzl", "voxel_py_test")

voxel_py_test(
    name = "utils_test",
    srcs = [
        "utils_test.py",
    ],
    deps = [
        requirement("sematic"),
        "//core/infra/sematic/shared:utils",
    ],
)
