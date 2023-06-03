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

from loguru import logger
from mcap_protobuf.reader import read_protobuf_messages

# adapted from: https://github.com/foxglove/mcap/blob/main/python/examples/protobuf/reader.py


def main(args: argparse.Namespace):
    """
    Reads a log file

    Args:
        args (argparse.Namespace): input argument namespace
    """
    logger.info(f"Reading example log from: {args.log}")
    for message in read_protobuf_messages(
        source=args.log, log_time_order=True
    ):
        logger.info(
            f"Topic: {message.topic} [{message.log_time}]: \n{message.proto_msg}"
        )


def get_args() -> argparse.Namespace:
    """
    Parses input commandline arguments

    Returns:
        argparse.Namespace: the parse commandline arguments
    """
    parser = argparse.ArgumentParser(
        description="Example MCAP logging playback"
    )
    parser.add_argument(
        "--log",
        type=str,
        required=True,
        help="The input log to read",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main(get_args())
