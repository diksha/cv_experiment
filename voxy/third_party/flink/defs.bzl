"""Definitions related to building flink pipelines and pipeline images"""

load("@aspect_bazel_lib//lib:utils.bzl", "to_label")
load("@bazel_skylib//lib:dicts.bzl", "dicts")
load("@io_bazel_rules_docker//container:container.bzl", _container = "container")
load("@io_bazel_rules_docker//lang:image.bzl", "app_layer")
load("@pip_deps//:requirements.bzl", "requirement")
load("@rules_py//py:defs.bzl", "py_binary")

def py_flink_pipeline(name, srcs = [], main = None, deps = [], data = []):
    """This rule builds a flink pipeline.

       Three targets will be produced, a py_binary that is runnable locally and
       two container images. One is a runnable form of the py_binary and the
       other is an image compatible with the flink-kubernetes-operator

       <name> - this is the runnable py_binary for this pipeline
       <name>.app-image - an intermediary python container image that is runnable
       <name>.image - a flink container image compatible with flink-kubernetes-operator

    Args:
        name (str): Pipeline name
        srcs (list, optional): Pipeline source files. Defaults to [].
        main (str, optional): Pipeline main file. Defaults to None.
        deps (list, optional): Pipeline deps. Defaults to [].
        data (list, optional): Pipeline data deps. Defaults to [].
    """
    py_binary(
        name = name,
        srcs = srcs,
        main = main,
        deps = deps + [
            requirement("apache-flink"),
            requirement("apache-flink-libraries"),
            requirement("setuptools"),
            "//:voxel_base_import",
        ],
        data = data,
    )

    app_layer_name = "{}.app-image".format(name)
    app_layer(
        name = app_layer_name,
        binary = to_label(name),
        base = "//third_party/flink:baseimage",
    )

    image_name = "{}.image".format(name)
    _py_flink_image(
        name = image_name,
        base = to_label(app_layer_name),
        binary = to_label(name),
    )

def _py_flink_image_impl(ctx):
    app_short_path = ctx.attr.binary[DefaultInfo].files_to_run.executable.short_path
    app_path = "/app/{}".format(app_short_path)
    runfiles_path = "{}.runfiles".format(app_path)

    return _container.image.implementation(
        ctx = ctx,
        env = dicts.add(ctx.attr.env, {
            "APP": app_path,
            "RUNFILES_DIR": runfiles_path,
        }),
    )

_py_flink_image = rule(
    attrs = dicts.add(_container.image.attrs, {
        "binary": attr.label(
            mandatory = True,
            cfg = "target",
        ),
    }),
    executable = True,
    outputs = _container.image.outputs,
    implementation = _py_flink_image_impl,
    toolchains = ["@io_bazel_rules_docker//toolchains/docker:toolchain_type"],
    cfg = _container.image.cfg,
)
