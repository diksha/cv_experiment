from argparse import ArgumentParser, Namespace
from dataclasses import dataclass

import awsiot.greengrasscoreipc
from awsiot.greengrasscoreipc.model import GetSecretValueRequest


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
    TIMEOUT = 10
    ipc_client = awsiot.greengrasscoreipc.connect()
    request = GetSecretValueRequest()
    request.secret_id = args.secret_name
    request.version_stage = "AWSCURRENT"
    operation = ipc_client.new_get_secret_value()
    operation.activate(request)
    futureResponse = operation.get_response()
    response = futureResponse.result(TIMEOUT)
    print(response.secret_value.secret_string)
