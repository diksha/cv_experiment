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

"""Update rules for creating py3 container image.
"""

load("@io_bazel_rules_docker//container:container.bzl", "container_image", _container = "container")
load("@io_bazel_rules_docker//lang:image.bzl", "app_layer")
load("@io_bazel_rules_docker//python:image.bzl", "py_layer")

def _python_binary_symlink_impl(ctx):
    # rstrip works on individual characters therefore use partition.
    # develop.py with rstrip .py becomes develo instead of develop.
    relative_path = ctx.file.binary.path.partition(".py")[0]
    symlink_path = "/".join([ctx.attr.directory, relative_path])
    return _container.image.implementation(
        ctx,
        symlinks = {
            symlink_path: "/app/third_party/containers/python3/common.runfiles/voxel/" + relative_path,
        },
    )

_python_binary_symlink_attrs = {
    "binary": attr.label(
        executable = True,
        allow_single_file = True,
        cfg = _container.image.cfg,
    ),
}
_python_binary_symlink_attrs.update(_container.image.attrs)

_python_binary_symlink = rule(
    attrs = _python_binary_symlink_attrs,
    executable = True,
    outputs = _container.image.outputs,
    toolchains = ["@io_bazel_rules_docker//toolchains/docker:toolchain_type"],
    implementation = _python_binary_symlink_impl,
)

def python_binary_symlink(name, directory, **kwargs):
    _python_binary_symlink(name = name, directory = directory, **kwargs)
    return name

def py3_image(name, base = None, layers = [], **kwargs):
    """py3_image constructs a container which can run a voxel python app

    Args:
        name (str): target name
        base (str, optional): Base image to use. Defaults to "@voxel_cuda_11_4_ubuntu_20_04//image".
        layers (list, optional): Additional layers to include in this image.
        **kwargs (dict, optional): Optional attributes commonly supported across most/all rules (like visibility, data, testonly).
    """
    base = base or "@voxel_cuda_11_4_ubuntu_20_04//image"
    binary_name = "//third_party/containers/python3:common"
    tags = kwargs.get("tags", None)

    binaries = kwargs.get("binaries", [])
    binary = kwargs.get("binary")
    if binary:
        binaries.append(binary)

    layer_name = name + "-py_binary_layers"
    py_layer(name = layer_name, deps = binaries)

    for index, dep in enumerate(layers + [":" + layer_name]):
        base = app_layer(name = "%s.%d" % (name, index), base = base, dep = dep, tags = tags)
        base = app_layer(name = "%s.%d-symlinks" % (name, index), base = base, dep = dep, binary = binary_name, tags = tags)

    for index, user_binary in enumerate(binaries):
        base = python_binary_symlink(name = name + ".python_binary_symlink-" + str(index), directory = "/app", base = base, binary = user_binary + ".py")

    visibility = kwargs.get("visibility", None)
    app_layer(
        name = "%s-py" % (name),
        base = base,
        binary = binary_name,
        visibility = visibility,
        tags = tags,
        args = kwargs.get("args"),
        data = kwargs.get("data"),
        testonly = kwargs.get("testonly"),
        create_empty_workspace_dir = True,
        compression_options = ["--fast"],
    )

    container_image(
        name = name,
        base = "%s-py" % (name),
        symlinks = {
            "/usr/bin/python": "/app/python_x86_64-unknown-linux-gnu/bin/python3",
            "/usr/bin/python3": "/app/python_x86_64-unknown-linux-gnu/bin/python3",
        },
        entrypoint = ["/usr/bin/python3"],
    )
