"""Macros related to AWS Lambda
"""

load("@aspect_bazel_lib//lib:utils.bzl", "to_label")
load("@com_github_ash2k_bazel_tools//multirun:def.bzl", "command", "multirun")
load("@io_bazel_rules_docker//container:container.bzl", "container_push")
load(":go.bzl", "go_aws_lambda_zip_impl")
load(":python.bzl", "py_aws_lambda_zip_impl", _py_aws_lambda_container_image = "py_aws_lambda_container_image")

def aws_lambda(name, function_name, zip_file = None, image = None, image_repository = None, image_registry = None):
    """Creates a binary target that when run will update lambda function code

    This supports either a zip file or a container image.

    The zip file must conform to the specification provided by AWS for whatever the
    Lambda's runtime is (Node, Go, Python, Custom). The container must do the same.

    Docs can be found here:

    https://docs.aws.amazon.com/lambda/latest/dg/welcome.html

    Args:
        name             (str): target name
        function_name    (str): lambda function name to target with updates
        zip_file         (str): function code zip file
        image          (Label): container_image target with the function code
        image_repository (str): image repository to push code to
        image_registry   (str): image registry to push code to (likely an ECR)
    """

    # maybe consider adding a lambda invoker here for debugging purposes

    if zip_file == None and image == None:
        fail("aws_lambda requires zip_file or image to be set")

    if zip_file != None and image != None:
        fail("aws_lambda requires only one of zip_file or image to be set, not both")

    if image and not (image_repository and image_registry):
        fail("aws_lambda image_repository and image_registry must be set for image")

    if image == None:
        command(
            name = name + ".update",
            command = "//third_party/aws/lambda:update-function-code",
            environment = {
                "FUNCTION_NAME": function_name,
                "ZIP_FILE": "$(rootpath {})".format(zip_file),
            },
            data = [zip_file],
        )
    else:
        tag_file = image + ".json.sha256"
        container_push(
            name = name + ".push",
            format = "Docker",
            image = image,
            registry = image_registry,
            repository = image_repository,
            tags = ["manual"],
            tag_file = tag_file,
        )

        command(
            name = name + ".update-function-code",
            command = "//third_party/aws/lambda:update-function-code",
            environment = {
                "FUNCTION_NAME": function_name,
                "IMAGE_REPO": "{}/{}".format(image_registry, image_repository),
                "IMAGE_TAG_FILE": "$(rootpath {})".format(tag_file),
            },
            data = [tag_file],
        )

        multirun(
            name = name + ".update",
            commands = [
                to_label(name + ".push"),
                to_label(name + ".update-function-code"),
            ],
        )

go_aws_lambda_zip = go_aws_lambda_zip_impl

def go_aws_lambda(name, function_name, target):
    """Creates a binary target that when run will update a Go lambda function from the passed in go_binary target.

    Go is the simplest to package, the zip package just has a go binary at the root of the zip file, with
    the lambda target name just being the name of the executable binary.

    Args:
        name          (str): lambda function name
        target      (Label): label pointing to a go_binary target
        function_name (str): lambda function name to target with updates
    """
    zip_name = "{}.artifact".format(name)
    go_aws_lambda_zip(
        name = zip_name,
        target = target,
    )

    aws_lambda(
        name = name,
        function_name = function_name,
        zip_file = zip_name + ".zip",
    )

# export py_aws_lambda_zip here
py_aws_lambda_zip = py_aws_lambda_zip_impl

def py_aws_lambda(name, function_name, target):
    """Creates a binary target that will update a python lambda from a py_binary target

    Args:
        name (str): lambda function name
        function_name (str): lambda function name to update
        target (Label): py_binary target to use as a target
    """
    zip_name = "{}.artifact".format(name)
    py_aws_lambda_zip(
        name = zip_name,
        target = target,
    )

    aws_lambda(
        name = name,
        function_name = function_name,
        zip_file = zip_name + ".zip",
    )

py_aws_lambda_container_image = _py_aws_lambda_container_image
