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

from copy import deepcopy

from scaleapi.exceptions import (
    ScaleInternalError,
    ScaleServiceUnavailable,
    ScaleTimeoutError,
)
from scaleapi.tasks import TaskReviewStatus, TaskStatus, TaskType

from core.labeling.scale.lib.scale_client import get_scale_client
from core.utils.voxel_decorators import retry_handler


class ScaleTaskWrapper:
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
    def get_single_task(self, task_id: str):
        """
        Get a scale task object given a task_id
        """
        return self._client.get_task(task_id)

    @retry_handler(
        exceptions=(
            ScaleInternalError,
            ScaleServiceUnavailable,
            ScaleTimeoutError,
        ),
        max_retry_count=2,
    )
    def _get_task_type_from_string(self, type_string: str):
        """
        Converts string to scale TaskType enum

        Args:
            type_string (str): annotation, categorization, comparison,
                               cuboidannotation, datacollection, documentmodel,
                               documenttranscription, imageannotation, laneannotation,
                               lidarannotation, lidarlinking, lidarsegmentation,
                               lidartopdown, lineannotation, namedentityrecognition,
                               pointannotation, polygonannotation, segmentannotation,
                               transcription, textcollection, videoannotation,
                               videoboxannotation, videoplaybackannotation, videocuboidannotation,
        """
        type_enum = None
        if type_string:
            type_enum = TaskType(type_string)
        return type_enum

    @retry_handler(
        exceptions=(
            ScaleInternalError,
            ScaleServiceUnavailable,
            ScaleTimeoutError,
        ),
        max_retry_count=2,
    )
    def _get_task_status_from_string(self, status_string: str):
        """
        Converts string to scale TaskStatus enum

        Args:
            status_string (str): pending, canceled, completed
        """
        status_enum = None
        if status_string:
            status_enum = TaskStatus(status_string)
        return status_enum

    @retry_handler(
        exceptions=(
            ScaleInternalError,
            ScaleServiceUnavailable,
            ScaleTimeoutError,
        ),
        max_retry_count=2,
    )
    def _get_task_review_status_from_string(self, review_status_string: str):
        """
        Converts string to scale TaskReviewStatus enum

        Args:
            review_status_string (str): accepted, fixed, commented, rejected
        """
        review_status_enum = None
        if review_status_string:
            review_status_enum = TaskReviewStatus(review_status_string)
        return review_status_enum

    @retry_handler(
        exceptions=(
            ScaleInternalError,
            ScaleServiceUnavailable,
            ScaleTimeoutError,
        ),
        max_retry_count=2,
    )
    def get_task_count(self, **kwargs):
        """
        Get a count of the number of tasks given kwarg filters.
        Filters that are not specified are not applied

        Args:
            kwargs:
                project_name (str): name of project to collect tasks
                batch_name (str): name of batch to collect tasks
                task_type (str): see _get_task_type_from_string
                status (str): see _get_task_status_from_string
                review_status (str): see _get_task_review_status_from_string
                unique_id (str): unique id of task
                completed_after (str): grabs tasks completed after date in format of %Y-%m-%d %h:%m:%s:%ms
                completed_before (str): grabs tasks completed before date in format of %Y-%m-%d %h:%m:%s:%ms
                updated_after (str): grabs tasks updated after date in format of %Y-%m-%d %h:%m:%s:%ms
                updated_before (str): grabs tasks updated before date in format of %Y-%m-%d %h:%m:%s:%ms
                created_after (str): grabs tasks created after date in format of %Y-%m-%d %h:%m:%s:%ms
                created_before (str): grabs tasks created before date in format of %Y-%m-%d %h:%m:%s:%ms
                tags: TODO

        Returns:
            count of tasks
        """
        return self._client.get_tasks_count(
            project_name=kwargs.get("project_name"),
            batch_name=kwargs.get("batch_name"),
            task_type=(
                self._get_task_type_from_string(kwargs.get("task_type"))
            ),
            status=self._get_task_status_from_string(kwargs.get("status")),
            review_status=(
                self._get_task_review_status_from_string(
                    kwargs.get("review_status")
                )
            ),
            unique_id=kwargs.get("unique_id"),
            completed_after=kwargs.get("completed_after"),
            completed_before=kwargs.get("completed_before"),
            updated_after=kwargs.get("updated_after"),
            updated_before=kwargs.get("updated_before"),
            created_after=kwargs.get("created_after"),
            created_before=kwargs.get("created_before"),
            tags=kwargs.get("tags"),
        )

    @retry_handler(
        exceptions=(
            ScaleInternalError,
            ScaleServiceUnavailable,
            ScaleTimeoutError,
        ),
        max_retry_count=2,
    )
    def active_task_generator(self, **kwargs):
        """
        Returns a generator of batches excluding inactive tasks given kwargs
        for filtering (see get_task_count)

        Args:
            kwargs: see get_tasks from scale

        Yields:
            Task: scale Task object
        """
        args = {
            "project_name": kwargs.get("project_name"),
            "batch_name": kwargs.get("batch_name"),
            "task_type": (
                self._get_task_type_from_string(kwargs.get("task_type"))
            ),
            "status": self._get_task_status_from_string(kwargs.get("status")),
            "review_status": (
                self._get_task_review_status_from_string(
                    kwargs.get("review_status")
                )
            ),
            "unique_id": kwargs.get("unique_id"),
            "completed_after": kwargs.get("completed_after"),
            "completed_before": kwargs.get("completed_before"),
            "updated_after": kwargs.get("updated_after"),
            "updated_before": kwargs.get("updated_before"),
            "created_after": kwargs.get("created_after"),
            "created_before": kwargs.get("created_before"),
            "tags": kwargs.get("tags"),
        }
        inactive_tags_args = deepcopy(args)
        inactive_tags_args["tags"] = (
            inactive_tags_args["tags"].append("inactive")
            if inactive_tags_args["tags"]
            else ["inactive"]
        )
        inactive_task_ids = [
            task.id for task in self._client.get_tasks(**inactive_tags_args)
        ]
        for task in self._client.get_tasks(**args):
            if task.id not in inactive_task_ids:
                yield task

    @retry_handler(
        exceptions=(
            ScaleInternalError,
            ScaleServiceUnavailable,
            ScaleTimeoutError,
        ),
        max_retry_count=2,
    )
    def get_active_tasks(self, **kwargs):
        """
        Returns a list of tasks excluding inactive tasks given kwargs
        for filtering (see get_task_count)
        """
        return list(self.active_task_generator(**kwargs))

    def get_task_id_from_unique_id(self, unique_id: str, project: str) -> str:
        """Get task ID from unique Scale task ID, used for cancelling task

        Args:
            unique_id (str): Scale task unique ID
            project (str): project to check task in

        Raises:
            Exception: raised if unique ID is not unique

        Returns:
            str: Scale task ID (can be used for cancellation)
        """
        tasks = list(
            self._client.get_tasks(project_name=project, unique_id=unique_id)
        )
        if len(tasks) != 1:
            raise Exception(
                "There is more than one task with the same unique id"
            )
        return tasks[0].task_id

    def inactivate_task(self, task_id: str):
        """Inactivate task in Scale

        Args:
            task_id (str): Scale task ID
        """
        self._client.set_task_tags(task_id, ["inactive"])
        self._client.clear_task_unique_id(task_id)
