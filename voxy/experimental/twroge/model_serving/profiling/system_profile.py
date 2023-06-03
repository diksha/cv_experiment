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
import signal

# trunk-ignore-all(bandit/B404,bandit/B603)
import subprocess
import typing
from datetime import datetime
from time import sleep, time_ns

import psutil
from loguru import logger
from mcap_protobuf.writer import Writer

# No name in module
# trunk-ignore(pylint/E0611)
from experimental.twroge.model_serving.profiling.proto.v1.profiling_pb2 import (
    GpuUsage,
    SystemUsage,
)

# adapted from: https://github.com/foxglove/mcap/blob/main/python/examples/protobuf/writer.py


def get_system_usage() -> typing.Tuple[SystemUsage, int]:
    """
    Grabs the system usage

    Returns:
        typing.Tuple[SystemUsage, int]: the system usage and the time in ns
    """

    mem = psutil.virtual_memory()
    mem_total = mem.total
    mem_used = mem.used
    mem_percent = mem.percent

    cpu_percent = psutil.cpu_percent()
    cpu_counts = [
        cpu_count for _, cpu_count in psutil.cpu_times()._asdict().items()
    ]
    usage = SystemUsage()
    usage.cpu_percent = cpu_percent
    usage.memory_percent = mem_percent
    usage.memory_total = mem_total
    usage.memory_used = mem_used
    usage.cpu_count_total = sum(cpu_counts)

    for item in psutil.cpu_times(percpu=True):
        cpu_counts = [cpu_count for _, cpu_count in item._asdict().items()]
        usage.cpu_core_counts.append(sum(cpu_counts))

    return usage, time_ns()


def get_gpu_usage() -> typing.Tuple[GpuUsage, int]:
    """
    Logs the gpu usage

    Returns:
        typing.Tuple[GpuUsage, int]: the gpu usage and the time in ns
    """
    command = [
        "nvidia-smi",
        "--query-gpu=timestamp,utilization.memory,utilization.gpu",
        "--format=csv",
    ]
    gpu_line = None
    gpu_memory = 0.0
    gpu_utilization = 0.0
    gpu_time_ns = 0
    with subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ) as process:
        process.stdout.readline()  # skip header
        gpu_line = process.stdout.readline()
    if gpu_line:
        timestamp, gpu_memory, gpu_utilization = gpu_line.decode(
            "utf-8"
        ).split(",")
        gpu_memory = float(gpu_memory.strip().split(" ")[0])
        gpu_utilization = float(gpu_utilization.strip().split(" ")[0])
        gpu_time_ns = int(
            datetime.strptime(
                timestamp.strip(), "%Y/%m/%d %H:%M:%S.%f"
            ).timestamp()
            * 1e9
        )
    return (
        GpuUsage(gpu_memory_percent=gpu_memory, gpu_percent=gpu_utilization),
        gpu_time_ns,
    )


SIGINT_RECEIVED = False


def handle_sigint(_signal, _frame):
    """
    Handles ctrl+c signal

    """
    # trunk-ignore-begin(pylint/W0603)
    global SIGINT_RECEIVED
    # trunk-ignore-end(pylint/W0603)
    SIGINT_RECEIVED = True
    logger.warning("Ctrl+c received. Exiting")


signal.signal(signal.SIGINT, handle_sigint)


def main(args: argparse.Namespace):
    """
    Generates an system profile log

    Args:
        args (argparse.Namespace): input argument namespace
    """
    logger.info(f"Writing log to: {args.output}")
    samples = 0
    with open(args.output, "wb") as out_file, Writer(out_file) as mcap_writer:
        while not SIGINT_RECEIVED:
            message, time = get_system_usage()
            mcap_writer.write_message(
                topic="/system_profile",
                message=message,
                log_time=time,
                publish_time=time,
            )
            gpu_message, gpu_time_ns = get_gpu_usage()
            mcap_writer.write_message(
                topic="/system_profile/gpu",
                message=gpu_message,
                log_time=gpu_time_ns,
                publish_time=gpu_time_ns,
            )
            # log at 1 hz
            sleep(1)
            samples += 1
    logger.success(
        f"Logging Complete. See log at {args.output}. Total Samples: {samples}"
    )


def get_args() -> argparse.Namespace:
    """
    Parses input commandline arguments

    Returns:
        argparse.Namespace: the parse commandline arguments
    """
    parser = argparse.ArgumentParser(
        description="System profiling MCAP logging"
    )
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
