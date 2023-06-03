load("@pip_deps//:requirements.bzl", "requirement")
load("@rules_python//python:defs.bzl", "py_library")

py_library(
    name = "resources",
    srcs = ["resources.py"],
    visibility = ["//visibility:public"],
    deps = [
        requirement("sematic"),
    ],
)

py_library(
    name = "aws_funcs",
    srcs = ["aws_funcs.py"],
    visibility = ["//visibility:public"],
    deps = [
        requirement("sematic"),
    ],
)

py_library(
    name = "mock_sematic_funcs",
    srcs = ["mock_sematic_funcs.py"],
    visibility = ["//visibility:public"],
    deps = [
        requirement("sematic"),
    ],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    visibility = ["//visibility:public"],
    deps = [
        requirement("loguru"),
        requirement("sematic"),
    ],
)
