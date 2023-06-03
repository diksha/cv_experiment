"""This module provides a rule for downloading artifacts from s3 using the bazel downloader
"""

load("@bazel_tools//tools/build_defs/repo:utils.bzl", "update_attrs", "workspace_and_buildfile")

def _get_signed_download_url(ctx, url):
    aws_cli = ctx.path(Label("@aws_cli//:aws/dist/aws"))
    cmd = [aws_cli, "s3", "presign", url]
    result = ctx.execute(cmd)

    if result.return_code != 0:
        fail("Failed to download {}: {}".format(url, result.stderr))

    return result.stdout

def _s3_archive_impl(ctx):
    download_url = _get_signed_download_url(ctx, ctx.attr.url)

    download_info = ctx.download_and_extract(
        url = download_url,
        output = ctx.attr.add_prefix,
        sha256 = ctx.attr.sha256,
        stripPrefix = ctx.attr.strip_prefix,
        type = ctx.attr.type,
    )
    workspace_and_buildfile(ctx)

    return update_attrs(ctx.attr, _s3_archive_attrs.keys(), {"sha256": download_info.sha256})

_s3_archive_attrs = {
    "add_prefix": attr.string(
        doc = "Prefix to add when archive is unpacked",
        default = "",
    ),
    "build_file": attr.label(
        allow_single_file = True,
        doc = "BULID.bazel file for the unpacked archive",
    ),
    "build_file_content": attr.string(
        doc = "The contents of the BUILD.bazel file for the target",
    ),
    "sha256": attr.string(
        doc = """The sha256 of the archive file being downloaded.

It is a security risk to leave this unspecified.
""",
    ),
    "strip_prefix": attr.string(
        doc = "Prefix to strip when archive is unpacked",
    ),
    "type": attr.string(
        doc = """The archive type of the downloaded file.

Valid types are `"zip"`, `"jar"`, `"war"`, `"aar"`, `"tar"`, `"tar.gz"`, `"tgz"`,
`"tar.xz"`, `"txz"`, `"tar.zst"`, `"tzst"`, `tar.bz2`, `"ar"`, or `"deb"`. 
""",
    ),
    "url": attr.string(
        mandatory = True,
        doc = "S3 url in the form of s3://<bucket>/<key>",
    ),
    "workspace_file": attr.label(
        allow_single_file = True,
        doc = "WORKSPACE file for the unpacked archive",
    ),
    "workspace_file_content": attr.string(
        doc = "WORKSPACE file contents for the unpacked archive",
    ),
}

s3_archive = repository_rule(
    implementation = _s3_archive_impl,
    attrs = _s3_archive_attrs,
)
