"""Definitions related to building flink pipelines and pipeline images"""

load("//third_party/flink:defs.bzl", "py_flink_pipeline")

def py_bowser_processor(name, main_src, deps = [], data = []):
    """This rule builds a bowser pipeline.

    Args:
        name (str): Pipeline name
        main_src (str): main source of a Flink Processor
        deps (list, optional): Pipeline deps. Defaults to [].
        data (list, optional): Pipeline data deps. Defaults to [].
    """
    py_flink_pipeline(
        name = name,
        srcs = [main_src],
        main = main_src,
        deps = deps + [
            "//services/platform/bowser/engine:bowser_engine",
            "//protos/platform/bowser/v1:v1_py_library",
            "@rules_python//python/runfiles",
        ],
        data = data + [
            "//configs",
        ],
    )
