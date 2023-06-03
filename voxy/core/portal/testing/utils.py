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
import random
import re
from datetime import datetime, timedelta

from django.utils import timezone


def random_datetime(start: datetime = None, end: datetime = None) -> datetime:
    """Return a random datetime between two datetime objects."""
    start = start or timezone.now() - timedelta(days=365 * 2)
    end = end or timezone.now()
    return start + timedelta(
        seconds=random.randint(0, int((end - start).total_seconds())),
    )


def strip_non_alphanumeric(value: str) -> str:
    return re.sub(r"\W+", "", value)
