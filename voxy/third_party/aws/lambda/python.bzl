"""Private implementation of py_aws_lambda_zip, do not use this use the reference in defs.bzl

Based on this stackoverflow answer with some substantial changes:

https://stackoverflow.com/questions/33569045/access-denied-using-boto3-through-aws-lambda

"""

load("@bazel_skylib//lib:dicts.bzl", "dicts")
load("@io_bazel_rules_docker//container:container.bzl", _container = "container")

def _contains(pattern):
    return "contains:" + pattern

def _startswith(pattern):
    return "startswith:" + pattern

def _endswith(pattern):
    return "endswith:" + pattern

def _is_ignored(path, patterns):
    for p in patterns:
        if p.startswith("contains:"):
            if p[len("contains:"):] in path:
                return True
        elif p.startswith("startswith:"):
            if path.startswith(p[len("startswith:"):]):
                return True
        elif p.startswith("endswith:"):
            if path.endswith(p[len("endswith:"):]):
                return True
        else:
            fail("Invalid pattern: " + p)

    return False

def _short_path(file_):
    # Remove prefixes for external and generated files.
    # E.g.,
    #   ../py_deps_pypi__pydantic/pydantic/__init__.py -> pydantic/__init__.py
    short_path = file_.short_path
    if short_path.startswith("../"):
        second_slash = short_path.index("/", 3)
        short_path = short_path[second_slash + 1:]

    if short_path.startswith("site-packages/"):
        short_path = short_path[len("site-packages/"):]

    return short_path

def _py_aws_lambda_file_map(srcs, deps):
    # ignore all these by default
    ignore = [
        # python bytecode cache files
        _contains("__pycache__"),
        _endswith(".pyc"),
        _endswith(".pyo"),
        # aws includes boto3
        _startswith("boto3/"),
        _startswith("botocore/"),
    ]

    file_map = {}

    for dep in deps.to_list():
        # this happens before the ignore list because
        # it is easiest to detect hermetic python from the
        # real shortpath rather than the stripped one
        if dep.short_path.startswith("../python_"):
            continue

        # create a stripped shortpath without `../pip_deps_...`
        short_path = _short_path(dep)

        # ignores the patterns listed above
        if _is_ignored(short_path, ignore):
            continue

        file_map[short_path] = dep

    # add our source files at the package root
    for source_file in srcs.to_list():
        file_map[source_file.basename] = source_file

    return file_map

def _py_aws_lambda_zip_impl(ctx):
    srcs = ctx.attr.target[DefaultInfo].files
    deps = ctx.attr.target[DefaultInfo].default_runfiles.files
    f = ctx.outputs.output
    file_map = _py_aws_lambda_file_map(srcs, deps)

    args = ctx.actions.args()
    args.use_param_file("@%s", use_always = True)
    args.set_param_file_format("multiline")

    for short_name in file_map:
        args.add(short_name + "=" + file_map[short_name].path)

    ctx.actions.run(
        outputs = [f],
        inputs = deps,
        executable = ctx.executable._zipper,
        arguments = ["cC", f.path, args],
        progress_message = "Creating archive...",
        mnemonic = "archiver",
    )

    out = depset(direct = [f])
    return [
        DefaultInfo(
            files = out,
        ),
        OutputGroupInfo(
            all_files = out,
        ),
    ]

_py_aws_lambda_zip = rule(
    implementation = _py_aws_lambda_zip_impl,
    attrs = {
        "output": attr.output(),
        "target": attr.label(),
        "_zipper": attr.label(
            default = Label("@bazel_tools//tools/zip:zipper"),
            cfg = "exec",
            executable = True,
        ),
    },
    executable = False,
    test = False,
)

def py_aws_lambda_zip_impl(name, target, **kwargs):
    _py_aws_lambda_zip(
        name = name,
        target = target,
        output = name + ".zip",
        **kwargs
    )

def _py_aws_lambda_container_layer_impl(ctx):
    srcs = ctx.attr.target[DefaultInfo].files
    deps = ctx.attr.target[DefaultInfo].default_runfiles.files
    file_map = _py_aws_lambda_file_map(srcs, deps)
    return _container.image.implementation(
        ctx,
        file_map = file_map,
        directory = "/var/task",
        workdir = "/var/task",
        cmd = [ctx.attr.handler],
    )

_py_aws_lambda_container_image = rule(
    implementation = _py_aws_lambda_container_layer_impl,
    attrs = dicts.add(_container.image.attrs, {
        "base": attr.label(
            default = Label("@voxel_lambda_python_baseimage//image"),
        ),
        "handler": attr.string(
            mandatory = True,
        ),
        "target": attr.label(
            mandatory = True,
            cfg = "target",
        ),
    }),
    outputs = _container.image.outputs,
    cfg = _container.image.cfg,
    toolchains = ["@io_bazel_rules_docker//toolchains/docker:toolchain_type"],
    executable = True,
)

def py_aws_lambda_container_image(name, target, handler):
    _py_aws_lambda_container_image(
        name = name,
        target = target,
        handler = handler,
    )
