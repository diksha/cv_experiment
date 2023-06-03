load("@pip_deps//:requirements.bzl", "requirement")
load("@rules_python//python:defs.bzl", "py_binary")

py_binary(
    name = "test",
    srcs = ["test.py"],
    data = [
        "@artifacts_doors_0630",
    ],
    deps = [
        requirement("attrs"),
        requirement("shapely"),
        requirement("cython"),
    ],
)
