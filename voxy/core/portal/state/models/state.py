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
from django.db.models import Q
from timescale.db.models.fields import TimescaleDateTimeField

from core.portal.state.models.abstract import AbstractModel
from core.structs.actor import ActorCategory
from core.structs.ergonomics import ActivityType, PostureType

actor_category_choices = [(item.value, item.name) for item in ActorCategory]
activity_type_choices = [(item.value, item.name) for item in ActivityType]
posture_type_choices = [(item.value, item.name) for item in PostureType]


class State(AbstractModel):
    class Meta:
        app_label = "state"
        indexes = [
            models.Index(
                name="timestamp_range_idx",
                fields=[
                    "timestamp",
                    "-end_timestamp",
                ],
            ),
            models.Index(
                fields=[
                    "organization",
                ]
            ),
            models.Index(
                fields=[
                    "location",
                ]
            ),
            models.Index(
                name="state_motion_zone_timestamps",
                fields=[
                    "actor_category",
                    "actor_id",
                    "timestamp",
                    "-end_timestamp",
                ],
                condition=Q(
                    actor_category=ActorCategory.MOTION_DETECTION_ZONE.value
                ),
            ),
        ]

    # Partioning column.
    timestamp = TimescaleDateTimeField(interval="1 day")

    camera_uuid = models.TextField()
    organization = models.TextField()
    location = models.TextField()
    zone = models.TextField()
    camera_name = models.TextField()

    actor_id = models.CharField(max_length=64, null=False)
    actor_category = models.IntegerField(choices=actor_category_choices)
    end_timestamp = models.DateTimeField()
    run_uuid = models.CharField(max_length=64, null=True)

    door_is_open = models.BooleanField(null=True)
    person_activity_type = models.IntegerField(
        choices=activity_type_choices, null=True
    )
    person_posture_type = models.IntegerField(
        choices=posture_type_choices, null=True
    )
    person_lift_type = models.IntegerField(
        choices=posture_type_choices, null=True
    )
    person_reach_type = models.IntegerField(
        choices=posture_type_choices, null=True
    )
    person_is_wearing_safety_vest = models.BooleanField(null=True)
    person_is_wearing_hard_hat = models.BooleanField(null=True)
    person_is_carrying_object = models.BooleanField(null=True)

    pit_is_stationary = models.BooleanField(null=True)
    person_is_associated = models.BooleanField(null=True)
    pit_in_driving_area = models.BooleanField(null=True)
    person_in_no_ped_zone = models.BooleanField(null=True)
    pit_is_associated = models.BooleanField(null=True)
    motion_zone_is_in_motion = models.BooleanField(null=True)
    num_persons_in_no_ped_zone = models.IntegerField(null=True)
