"""Pytest macro

This macro simplifies generating a test which reports coverage correctly.
"""

load("@pip_deps//:requirements.bzl", "requirement")
load("@rules_py//py:defs.bzl", "py_test")

def voxel_py_test(name, srcs, deps = [], args = [], **kwargs):
    """Wrapper around py_test to ensure pytest + Bazel play nicely together."""
    py_test(
        name = name,
        srcs = [
            "//third_party/pytest:wrapper.py",
        ] + srcs,
        main = "//third_party/pytest:wrapper.py",
        args = args + [
            "-p third_party.pytest.coverage_plugin",
            # Show 10 slowest tests
            "--durations=10",
        ] + ["$(location :%s)" % x for x in srcs],
        deps = deps + [
            requirement("mock"),
            requirement("pytest"),
            requirement("pytest-mock"),
            "//third_party/pytest:coverage_plugin",
            "//:voxel_base_import",
        ],
        **kwargs
    )
