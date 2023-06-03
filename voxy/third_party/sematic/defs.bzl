"""Bazel macros for usage in Voxel's Sematic deployment"""

load("@rules_sematic//:pipeline.bzl", "sematic_pipeline")
load("//:constants.bzl", "container_registry_for_sematic_push")

def voxel_sematic_pipeline(
        name,
        deps,
        registry = container_registry_for_sematic_push,
        repository = "sematic",
        data = None,
        base = "//third_party/sematic:no_cuda_base_image",
        bases = None,
        image_layers = None,
        env = None,
        **kwargs):
    """A wrapper to set more useful defaults for Voxel's Sematic targets.

    Args:
        name: name of the target
        deps: list of dependencies
        registry: (optional) URI of the container registry to use to register
            the container image
        repository: (optional) container repository for the image
        data: (optional) data files to add to the image
        base: (optional) label of the base image to use. Defaults to
            "//third_party/sematic:no_cuda_base_image". For an image that can
            use GPUs, use "//third_party/sematic:cuda_base_image"
        bases: (optional) map from a string for a custom base image name to the
            label of the base image to use. Ex:
            {
                "default": "//third_party/sematic:no_cuda_base_image",
                "cuda": "//third_party/sematic:cuda_base_image"
            }
        image_layers: (optional) pass through arg to the `layers`
            parameter of `py3_image`: https://github.com/bazelbuild/rules_docker#py3_image
        env: (optional) mapping of environment variables to set in the container
        **kwargs:  extra keyword arguments to pass to sematic_pipeline.
    """
    sematic_pipeline(
        name = name,
        deps = deps,
        registry = registry,
        repository = repository,
        data = data,
        base = base,
        bases = bases,
        image_layers = image_layers,
        env = env,
        **kwargs
    )
