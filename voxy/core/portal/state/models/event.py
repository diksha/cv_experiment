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

from django.db import models
from timescale.db.models.fields import TimescaleDateTimeField

from core.portal.state.models.abstract import AbstractModel
from core.structs.event import EventType

event_type_choices = [(item.value, item.name) for item in EventType]


class Event(AbstractModel):
    class Meta:
        app_label = "state"

    # Partioning column.
    timestamp = TimescaleDateTimeField(interval="1 day")

    camera_uuid = models.TextField()
    organization = models.TextField()
    location = models.TextField()
    zone = models.TextField()
    camera_name = models.TextField()

    subject_id = models.CharField(max_length=64, null=True)
    object_id = models.CharField(max_length=64, null=True)
    event_type = models.IntegerField(choices=event_type_choices)

    end_timestamp = models.DateTimeField()
    run_uuid = models.CharField(max_length=64, null=True)
    x_velocity_pixel_per_sec = models.FloatField(null=True)
    y_velocity_pixel_per_sec = models.FloatField(null=True)
    normalized_speed = models.FloatField(null=True)
