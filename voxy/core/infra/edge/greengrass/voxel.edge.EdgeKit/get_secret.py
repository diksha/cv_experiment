import argparse
from dataclasses import dataclass

from mypy_boto3_secretsmanager.type_defs import SecretListEntryTypeDef

from core.utils.aws_utils import (
    get_secret_from_aws_secret_manager,
    list_secrets,
)

ALLOW_TAG_KEYS = ["edge:allowed-uuid:primary", "edge:allowed-uuid:secondary"]


@dataclass
class Args(argparse.Namespace):
    def __init__(self) -> None:
        super().__init__()
        self.thing_name: str = ""
        self.secret_suffix: str = ""


def parse_args() -> Args:
    parser = argparse.ArgumentParser()
    parser.add_argument("--thing_name", type=str)
    parser.add_argument("--secret_suffix", type=str)
    return parser.parse_args(namespace=Args())


def get_secret_arn(thing_name: str, secret_suffix: str) -> str:
    """Get the ARN for a secret matching a suffix + allowed edge device.

    Most secret names will have a customer-specific prefix, followed by
    a more generic suffix. For example, a `manifest.yaml` file might be
    stored in secrets for three different edge devices like this:

        voxel/server-1/manifest.yaml
        buzz/bazz/manifest.yaml
        foo/bar/manifest.yaml

    Each edge device should only be associated with one suffix at a time.
    The logic here is a little clunky because boto3 doesn't currently support
    AND operations in filters, so we can't filter with something like:

        WHERE key=foo AND value=bar

    it will actually behave like:

        WHERE key=foo OR value=bar

    So we instead get all secrets where the thing name is a tag value, then
    do client-side filtering to make sure one and only one secret has an
    "allow" tag for that device:

        edge:allowed-uuid:primary = thing_name

    Args:
        thing_name (str): Thing name/ID provided in the request (UUID4)
        secret_suffix (str): Suffix of the desired secret name

    Raises:
        RuntimeError: Raises if no secrets match or multiple secrets match

    Returns:
        str: The secret ARN
    """
    # Get all secrets where this thing name is a tag value
    secret_list = list_secrets(
        filters=[
            {
                "Key": "tag-value",
                "Values": [
                    thing_name,
                ],
            },
        ],
    )

    # Filter where tag key is an "allow" key and name ends with secret_suffix
    def secret_matches_suffix(secret: SecretListEntryTypeDef) -> bool:
        if not secret["Name"].endswith(secret_suffix):
            return False
        for tag in secret["Tags"]:
            if tag["Key"] in ALLOW_TAG_KEYS and tag["Value"] == thing_name:
                return True
        return False

    secrets = list(filter(secret_matches_suffix, secret_list))

    if len(secrets) > 1:
        raise RuntimeError(
            f"Multiple secrets found for thing_name: {thing_name}"
        )
    if len(secrets) == 0:
        raise RuntimeError(f"No secrets found for thing_name: {thing_name}")
    return secrets[0]["ARN"]


def print_value(thing_name: str, secret_suffix: str) -> None:
    secret_arn = get_secret_arn(thing_name, secret_suffix)
    secret_value = get_secret_from_aws_secret_manager(secret_arn)
    print(secret_value)


if __name__ == "__main__":
    args = parse_args()
    print_value(args.thing_name, args.secret_suffix)
