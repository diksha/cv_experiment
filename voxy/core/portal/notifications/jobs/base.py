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
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List

from core.portal.lib.jobs.base import JobBase
from core.portal.notifications.enums import NotificationCategory


class NotificationJob(JobBase, ABC):
    @property
    @abstractmethod
    def notification_category(self) -> NotificationCategory:
        """Notification category."""


class ScheduledNotificationJob(NotificationJob, ABC):
    @classmethod
    @abstractmethod
    def get_jobs_to_run(
        cls, invocation_timestamp: datetime = None, base_url: str = None
    ) -> List[NotificationJob]:
        """Returns list of jobs to run at the invocation timestamp.

        Args:
            invocation_timestamp (datetime): timestamp of when function was invoked.
            base_url (str): Base URL to use in content links.

        Returns:
            List of ready-to-run job instances.
        """
