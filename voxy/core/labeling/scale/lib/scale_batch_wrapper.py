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

from loguru import logger
from scaleapi.batches import BatchStatus
from scaleapi.exceptions import (
    ScaleInternalError,
    ScaleServiceUnavailable,
    ScaleTimeoutError,
)

from core.labeling.scale.lib.scale_client import get_scale_client
from core.utils.voxel_decorators import retry_handler


class ScaleBatchWrapper:
    def __init__(self, credentials_arn: str):
        self._client = get_scale_client(credentials_arn)

    @retry_handler(
        exceptions=(
            ScaleInternalError,
            ScaleServiceUnavailable,
            ScaleTimeoutError,
        ),
        max_retry_count=2,
    )
    def _get_batch_status_from_string(self, status_string: str):
        """
        Converts string to scale BatchStatus enum

        Args:
            status_string (str): staging, in_progress, completed
        """
        status_enum = None
        if status_string:
            status_enum = BatchStatus(status_string)
        return status_enum

    @retry_handler(
        exceptions=(
            ScaleInternalError,
            ScaleServiceUnavailable,
            ScaleTimeoutError,
        ),
        max_retry_count=2,
    )
    def get_batch_status(self, batch_name: str):
        """
        Get the status of a batch given the batch_name
        """
        return self._client.batch_status(batch_name)

    @retry_handler(
        exceptions=(
            ScaleInternalError,
            ScaleServiceUnavailable,
            ScaleTimeoutError,
        ),
        max_retry_count=2,
    )
    def get_batch(self, batch_name: str):
        """
        Get a scale batch object given a batch_name
        """
        return self._client.get_batch(batch_name)

    @retry_handler(
        exceptions=(
            ScaleInternalError,
            ScaleServiceUnavailable,
            ScaleTimeoutError,
        ),
        max_retry_count=2,
    )
    def cancel_batch(self, batch_name):
        batch = self.get_batch(batch_name)
        for task in self._client.get_tasks(
            project_name=batch.project, batch_name=batch.name
        ):
            self._client.cancel_task(task.id, True)
        logger.info(f"Canceled tasks and cleared unique id for {batch_name}")

    @retry_handler(
        exceptions=(
            ScaleInternalError,
            ScaleServiceUnavailable,
            ScaleTimeoutError,
        ),
        max_retry_count=2,
    )
    def batch_generator(self, **kwargs):
        """
        Returns a generator of batches given kwargs for filtering.
        Filters that are not specified are not applied

        Args:
            kwargs:
                project_name (str): name of project to collect batches
                batch_status (str): batch status corresponding to BatchStatus enum (see _get_batch_status_from_string)
                created_after (str): grabs batches created after date in format of %Y-%m-%d
                created_before (str): grabs batches created before date in format of %Y-%m-%d

        Returns:
            Generator of batches
        """
        return self._client.get_batches(
            project_name=kwargs.get("project_name"),
            batch_status=(
                self._get_batch_status_from_string(kwargs.get("batch_status"))
            ),
            created_after=kwargs.get("created_after"),
            created_before=kwargs.get("created_before"),
        )

    @retry_handler(
        exceptions=(
            ScaleInternalError,
            ScaleServiceUnavailable,
            ScaleTimeoutError,
        ),
        max_retry_count=2,
    )
    def get_batches(self, **kwargs):
        """
        Returns a list of batches given kwargs for filtering (see batch_generator)
        """
        return list(self.batch_generator(**kwargs))

    def get_completed_batches(
        self,
        task_list,
    ) -> list:
        """
        Get a list of completed batches that have been completed / updated between two dates

        Args:
            task_list (List): earliest date batch can be completed or updated to count
                for label conversion

        Returns:
            batches (list): list of batches to run label conversion
        """
        completed_batches = set()
        batch_status_map = {}
        for task in task_list:
            # TODO(diksha): Remove try catch after scale resolves batch error
            try:
                batch_name = task.batch
            except AttributeError:
                logger.error(f"Batch not found for task {task.id}")
                continue
            if not batch_status_map.get(batch_name):
                batch_status = self.get_batch_status(batch_name)["status"]
                batch_status_map[batch_name] = batch_status
                if batch_status == "completed":
                    completed_batches.add(batch_name)

        batches = list(completed_batches)
        return batches
