"""
    This module contains a release macro for the transcoder service.
"""

load("@com_github_ash2k_bazel_tools//multirun:def.bzl", "multirun")
load(
    "@io_bazel_rules_docker//container:container.bzl",
    "container_push",
)
load("//third_party/aws/greengrass:defs.bzl", "greengrass_component")

def transcoder_release(name, transcode_mode, environment, version, artifact):
    """Creates a transcoder release

    Args:
        name (str): Target name for this release
        transcode_mode (str): quicksync or cuda
        environment (str): prod or dev
        version (str): Greengrass component version (must be in semantic X.Y.Z format)
        artifact (label): Artifact zip file used by the application
    """

    if environment == "production":
        component_name_suffix = ""
    elif environment == "development":
        component_name_suffix = "Dev"
    else:
        fail("environment must be production or development")

    if transcode_mode == "quicksync":
        component_name_prefix = "Quicksync"
    elif transcode_mode == "cuda":
        component_name_prefix = "Cuda"
    else:
        fail("transcode_mode must be quicksync or cuda")

    component_name = "voxel.edge.{}Transcoder{}".format(component_name_prefix, component_name_suffix)
    edgeconfig_component_name = "voxel.edge.EdgeConfig{}".format(component_name_suffix)
    image_tag = "{}-{}".format(component_name, version)

    greengrass_component(
        name = name + ".greengrass",
        artifact = artifact,
        component_name = component_name,
        recipe_template = "transcoder-recipe.yaml",
        substitutions = {
            "$$COMPONENT_NAME$$": component_name,
            "$$EDGECONFIG_COMPONENT_NAME$$": edgeconfig_component_name,
            "$$IMAGE_TAG$$": image_tag,
            "$$TRANSCODE_MODE$$": transcode_mode,
        },
        version = version,
    )

    image = "//services/edge/transcoder/container:{}".format(transcode_mode)

    container_push(
        name = name + ".container.push",
        format = "Docker",
        image = image,
        registry = "360054435465.dkr.ecr.us-west-2.amazonaws.com",
        repository = "voxel/edge/edge-transcoder-" + transcode_mode,
        tag = image_tag,
        tags = ["manual"],
    )

    multirun(
        name = name + ".release",
        commands = [
            ":{}.container.push".format(name),
            ":{}.greengrass.create-version".format(name),
        ],
    )
