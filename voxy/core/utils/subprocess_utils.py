#
# Copyright 2022 Voxel Labs, Inc.
# All rights reserved.
#
# This document may not be reproduced, republished, distributed, transmitted,
# displayed, broadcast or otherwise exploited in any manner without the express
# prior written permission of Voxel Labs, Inc. The receipt or possession of this
# document does not convey any rights to reproduce, disclose, or distribute its
# contents, or to manufacture, use, or sell anything that it may describe, in
# whole or in part.
#

# trunk-ignore(bandit/B404)
import subprocess
import threading

# trunk-ignore(bandit/B404)
from subprocess import CalledProcessError
from typing import IO, Any, Callable, Dict, List

from loguru import logger


def log_stream(logger_func: Callable[[str], Any], stream: IO[bytes]):
    """Function to log stream

    Args:
        logger_func (Callable[str]): Function to log a message
        stream (IO[bytes]): stream to log
    """
    for line in stream:
        logger_func(line.decode("utf8").rstrip())


def logged_subprocess_call(
    cmd: List[str], checked: bool = False, **kwargs: Dict[str, Any]
) -> int:
    """Call subprocess and log output

    Args:
        cmd (List[str]): Command to run
        checked (bool, optional): Whether to throw exception on failure. Defaults to False.
        **kwargs (Dict[str, Any]): Other keyword arguments to create_subprocess_exec

    Raises:
        CalledProcessError: checked=True and process exited with non-zero status

    Returns:
        int: Exit status from subprocess
    """
    # trunk-ignore(bandit/B603)
    with subprocess.Popen(
        cmd, **kwargs, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ) as proc:
        threads = [
            threading.Thread(
                target=log_stream, args=(logger.info, proc.stdout)
            ),
            threading.Thread(
                target=log_stream, args=(logger.error, proc.stderr)
            ),
        ]
        for thread in threads:
            thread.start()
        status = proc.wait()
        for thread in threads:
            thread.join()

    if checked and status != 0:
        raise CalledProcessError(status, cmd)
    return status
