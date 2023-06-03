""" This module contains the greengrass_component macro used to set up new greengrass components
"""

load("@aspect_bazel_lib//lib:expand_make_vars.bzl", "expand_template")
load("@com_github_ash2k_bazel_tools//multirun:def.bzl", "command")

def greengrass_component(name, component_name, version, recipe_template, artifact, substitutions = {}, visibility = []):
    """This macro creates a greengrass component, allowing for some macro substitutions.

    To use this macro to create a greengrass component, run the target:
        {name}.create-version

    The following substitutions are built in:
        $$COMPONENT_NAME$$
        $$COMPONENT_VERSION$$
        $$COMPONENT_ARTIFACT_URI$$

    Args:
        name (str): A name for this target
        component_name (str): The greengrass component name
        version (str): Component version, like "0.0.1"
        recipe_template (str): Recipe template, filename or label.
        artifact (str): Artifact to ship with this component, will be uploaded to the s3 uri in $$COMPONENT_ARTIFACT_URI$$
        substitutions (dict, optional): Additional substitutions in this recipe file. Defaults to {}.
        visibility (list): Visibility to pass along.
    """
    artifact_uri = "s3://voxel-infra-edge/greengrass/{0}/{1}/artifact.zip".format(component_name, version)
    all_substitutions = {
        "$$COMPONENT_ARTIFACT_URI$$": artifact_uri,
        "$$COMPONENT_NAME$$": component_name,
        "$$COMPONENT_VERSION$$": version,
    }
    all_substitutions.update(substitutions)

    expand_template(
        name = name + ".recipe",
        out = component_name + ".recipe.yaml",
        substitutions = all_substitutions,
        template = recipe_template,
        visibility = visibility,
    )

    native.filegroup(
        name = name,
        srcs = [
            ":" + name + ".recipe",
            artifact,
        ],
        visibility = visibility,
    )

    command(
        name = name + ".create-version",
        environment = {
            "AWS_PROFILE": "production-admin",
            "COMPONENT_ARTIFACT_URI": artifact_uri,
            "COMPONENT_NAME": component_name,
            "COMPONENT_VERSION": version,
        },
        arguments = [
            "$(rootpath {})".format(component_name + ".recipe.yaml"),
            "$(rootpath {})".format(artifact),
        ],
        data = [
            component_name + ".recipe.yaml",
            artifact,
            "//third_party/aws",
        ],
        command = "//third_party/aws/greengrass:create-version".format(name),
        visibility = visibility,
    )
