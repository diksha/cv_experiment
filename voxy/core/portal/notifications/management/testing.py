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
from datetime import datetime

import pytz
from notifications.jobs.organization_daily_summary import (
    OrganizationDailySummaryJob,
)


def send_summary_email(
    to: str, organization_id: int, base_url: str = "http://localhost:9000"
):
    invocation_timestamp = datetime.utcnow().astimezone(
        pytz.timezone("US/Pacific")
    )

    job = OrganizationDailySummaryJob(
        organization_id=organization_id,
        invocation_timestamp=invocation_timestamp,
        base_url=base_url,
    )
    data = job.get_data()
    job.send_email(to, data)
