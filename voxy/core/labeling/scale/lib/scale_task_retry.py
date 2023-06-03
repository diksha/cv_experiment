#
# Copyright 2020-2021 Voxel Labs, Inc.
# All rights reserved.
#
# This document may not be reproduced, republished, distributed, transmitted,
# displayed, broadcast or otherwise exploited in any manner without the express
# prior written permission of Voxel Labs, Inc. The receipt or possession of this
# document does not convey any rights to reproduce, disclose, or distribute its
# contents, or to manufacture, use, or sell anything that it may describe, in
# whole or in part.
#

import time

# NOTE: This is intended to be removed long term but is here
#       in the short term to address the fragile scale task api
#       As these get updated and the scale reliability improves,
#       then this file should get deprecated
#
#       Please update with more issues as they come
#
#       Some open issues:
#         1. the scale python SDK sometimes retries to make a task on an error, so we have to go in
#            and delete the task, then recreate it.
#         2. The scale task api sometimes returns server errors that cause the pipelines to break
from typing import Callable

from loguru import logger
from scaleapi.exceptions import (
    ScaleDuplicateResource,
    ScaleInternalError,
    ScaleServiceUnavailable,
    ScaleTimeoutError,
)


# TODO: REMOVE ME HACK: please see note above
class ScaleTaskRetryWrapper:
    def __init__(
        self,
        task_creation_call: Callable,
        task_cancel_call: Callable,
        max_retries: int = 4,
        backoff_factor: int = 2,
        backoff_max_sleep_s: int = 60,
    ):
        self.task_creation_call = task_creation_call
        self.task_cancel_call = task_cancel_call
        self.server_errors = [
            ScaleInternalError,
            ScaleServiceUnavailable,
            ScaleTimeoutError,
        ]
        self.invalid_id_errors = [ScaleDuplicateResource]
        self.timeout_s = 1
        self.timeout_factor = backoff_factor
        self.timeout_multiplier = 1
        self.max_sleep_s = backoff_max_sleep_s  # 1 minute
        self.max_retries = max_retries

    def create_task(self):
        tries = 0
        while tries < self.max_retries:
            try:
                self.task_creation_call()
                return
            except tuple(self.invalid_id_errors) as invalid_id_error:
                logger.error(f"Encountered: {invalid_id_error}")
                logger.error("Retrying after cancelling task")
                try:
                    self.task_cancel_call()
                except Exception as e:  # trunk-ignore(pylint/W0703)
                    logger.error(f"Task could not be cancelled:  {e}")

                error = invalid_id_error
            except tuple(self.server_errors) as server_error:
                logger.error(f"Encountered: {server_error}")
                logger.error("Retrying ...")
                error = server_error
            sleep_time = min(
                self.max_sleep_s, self.timeout_s * self.timeout_multiplier
            )
            self.timeout_multiplier *= self.timeout_factor
            logger.warning(
                f"Encountered error, retrying after {sleep_time} seconds"
            )
            time.sleep(sleep_time)
            tries += 1
        logger.error("Max retries exceeded")
        logger.error(error)
        raise error
