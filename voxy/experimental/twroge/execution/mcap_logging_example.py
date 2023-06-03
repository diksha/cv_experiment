##
## Copyright 2022 Voxel Labs, Inc.
## All rights reserved.
##
## This document may not be reproduced, republished, distributed, transmitted,
## displayed, broadcast or otherwise exploited in any manner without the express
## prior written permission of Voxel Labs, Inc. The receipt or possession of this
## document does not convey any rights to reproduce, disclose, or distribute its
## contents, or to manufacture, use, or sell anything that it may describe, in
## whole or in part.
##

import argparse

import numpy as np
from loguru import logger
from mcap_protobuf.writer import Writer
from tqdm import tqdm

# No name in module
# trunk-ignore(pylint/E0611)
from experimental.twroge.execution.proto.v1.example_message_pb2 import (
    ExampleMessage,
)

# adapted from: https://github.com/foxglove/mcap/blob/main/python/examples/protobuf/writer.py


def main(args: argparse.Namespace):
    """
    Generates an example log file with the custom protobuf message

    Args:
        args (argparse.Namespace): input argument namespace
    """
    logger.info(f"Writing example log to: {args.output}")
    with open(args.output, "wb") as out_file, Writer(out_file) as mcap_writer:
        period = 2 * np.pi / 100
        for i in tqdm(range(1, 1000)):
            signal = np.sin(period * i) * 5
            example_message = ExampleMessage(
                field_a=f"Field A {i}",
                field_b=f"Field B {i}",
                counter=i,
                random_number=signal,
            )
            mcap_writer.write_message(
                topic="/example_messages/a",
                message=example_message,
                log_time=i * 1000000,
                publish_time=i * 1000000,
            )
            noise = np.random.normal(0, 1, size=(1))[0]
            example_message = ExampleMessage(
                field_a=f"Field A {i}",
                field_b=f"Field B {i}",
                counter=i,
                random_number=signal + noise,
            )
            mcap_writer.write_message(
                topic="/example_messages/b",
                message=example_message,
                log_time=i * 1000000,
                publish_time=i * 1000000,
            )


def get_args() -> argparse.Namespace:
    """
    Parses input commandline arguments

    Returns:
        argparse.Namespace: the parse commandline arguments
    """
    parser = argparse.ArgumentParser(description="Example MCAP logging")
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="The output file to write",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main(get_args())
