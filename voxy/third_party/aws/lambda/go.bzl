"""Private implementation of py_aws_lambda_zip, do not use this use the reference in defs.bzl

Based on this stackoverflow answer with some substantial changes:

https://stackoverflow.com/questions/33569045/access-denied-using-boto3-through-aws-lambda

"""

load("@aspect_bazel_lib//lib:paths.bzl", "to_manifest_path")
load("@io_bazel_rules_go//go:def.bzl", "GoArchive")

def _go_aws_lambda_zip_impl(ctx):
    target_name = ctx.attr.target[GoArchive].data.label.name
    prefix = "{}.runfiles".format(target_name)

    file_map = {}
    for file in ctx.attr.target[GoArchive].runfiles.files.to_list():
        file_map["{}/{}".format(prefix, to_manifest_path(ctx, file))] = file

    for file in ctx.attr.target[DefaultInfo].files.to_list():
        file_map[file.basename] = file

    args = ctx.actions.args()
    args.use_param_file("@%s", use_always = True)
    args.set_param_file_format("multiline")

    for short_name, file in file_map.items():
        args.add(short_name + "=" + file.path)

    f = ctx.outputs.output
    ctx.actions.run(
        outputs = [f],
        inputs = [file for file in file_map.values()],
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

_go_aws_lambda_zip = rule(
    implementation = _go_aws_lambda_zip_impl,
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

def go_aws_lambda_zip_impl(name, target, **kwargs):
    _go_aws_lambda_zip(
        name = name,
        target = target,
        output = name + ".zip",
        **kwargs
    )
