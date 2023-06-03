"""This module provides a macro for creating a triton model repository"""

load("@com_github_ash2k_bazel_tools//multirun:def.bzl", "command")

def triton_model_repository(name, config, repo_path, data):
    command(
        name = name + ".update",
        arguments = [
            "-config",
            "$(location {})".format(config),
            "-repo-path",
            repo_path,
        ],
        command = "//experimental/jorge/modelrepo/cmd/buildrepo",
        data = [
            config,
        ] + data,
    )
