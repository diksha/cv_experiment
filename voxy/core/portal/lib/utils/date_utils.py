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


def convert_tz(current_time: datetime, from_tz: str, to_tz: str) -> datetime:
    try:
        from_datetime = current_time.replace(tzinfo=pytz.timezone(from_tz))
        to_datetime = from_datetime.astimezone(pytz.timezone(to_tz))
        return to_datetime
    except pytz.exceptions.UnknownTimeZoneError:
        return current_time


def timezone_offset_hours(timezone: str) -> float:
    offset = datetime.now(pytz.timezone(timezone)).utcoffset()
    if offset:
        return offset.total_seconds() / 60.0 / 60.0
    return 0.0


def hours_between(start: datetime, end: datetime) -> float:
    duration = end - start
    return duration.total_seconds() / 60.0 / 60.0
