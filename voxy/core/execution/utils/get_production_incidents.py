#
# Copyright 2023 Voxel Labs, Inc.
# All rights reserved.
#
# This document may not be reproduced, republished, distributed, transmitted,
# displayed, broadcast or otherwise exploited in any manner without the express
# prior written permission of Voxel Labs, Inc. The receipt or possession of this
# document does not convey any rights to reproduce, disclose, or distribute its
# contents, or to manufacture, use, or sell anything that it may describe, in
# whole or in part.
#
import datetime
import typing

from core.structs.incident import Incident


def get_production_incidents(
    camera_uuid: str,
    start_date: datetime.datetime,
    end_date: datetime.datetime,
) -> typing.List[Incident]:
    """
    Gathers the production incidents from portal for the start time and end time

    TODO: get the production incidents via a portal session

    Args:
        camera_uuid (str): the camera uuid to grab incidents from
        start_date (datetime.datetime): the start date
        end_date (datetime.datetime): the end date

    Returns:
        typing.List[Incident]: the list of incidents generated in production
    """
    return []
