from argparse import ArgumentParser, Namespace
from dataclasses import dataclass

import boto3


@dataclass
class Args(Namespace):
    def __init__(self) -> None:
        super().__init__()
        self.timestamp: str = None


def parse_args() -> Args:
    parser = ArgumentParser()
    parser.add_argument("--timestamp", type=str)
    return parser.parse_args(namespace=Args())


if __name__ == "__main__":
    args = parse_args()
    s3 = boto3.client("s3")
    response = s3.get_object(
        Bucket="voxel-infra-edge",
        Key="greengrass/voxel.edge.EdgeKit/debug.zip",
    )

    print("======== DEBUGGING OUTPUT START ========")
    print(response)
    print("======== DEBUGGING OUTPUT END ========")
