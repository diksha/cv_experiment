"""Macro that can be used to run unit tests that depend on Django.
"""

load("@bazel_skylib//lib:new_sets.bzl", "sets")
load("@pip_deps//:requirements.bzl", "requirement")
load("//third_party/pytest:build_defs.bzl", "voxel_py_test")

def portal_django_test(*args, **kwargs):
    """Use this macro for portal unit tests which depend on Django.

    Args:
        *args: These arguments will be forwarded to voxel_py_test
        **kwargs: These arguments will be forwarded to voxel_py_test
    """

    # Override env
    env = kwargs.pop("env", {})
    env["ENVIRONMENT"] = "test"
    env["DJANGO_SETTINGS_MODULE"] = "core.portal.voxel.settings"

    # Override deps
    deps = kwargs.pop("deps", [])
    deps = sets.make(deps + [
        requirement("pytest-django"),
        requirement("factory_boy"),
        requirement("faker"),
        "//core/portal:app_lib",
    ])

    voxel_py_test(
        env = env,
        deps = sets.to_list(deps),
        args = kwargs.pop("args", []) + ["--import-mode=importlib"],
        *args,
        **kwargs
    )
