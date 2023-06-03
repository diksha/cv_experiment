from argparse import ArgumentParser, Namespace
from dataclasses import dataclass

import boto3


@dataclass
class Args(Namespace):
    def __init__(self) -> None:
        super().__init__()
        self.secret_name: str = None
        self.timestamp: str = None


def parse_args() -> Args:
    parser = ArgumentParser()
    parser.add_argument("--secret_name", type=str, required=True)
    parser.add_argument("--timestamp", type=str)
    return parser.parse_args(namespace=Args())


if __name__ == "__main__":
    args = parse_args()
    secrets_manager_client = boto3.client("secretsmanager")
    kwargs = {"SecretId": args.secret_name}
    response = secrets_manager_client.get_secret_value(**kwargs)
    print(response["SecretString"])
