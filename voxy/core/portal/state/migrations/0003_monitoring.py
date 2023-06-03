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

import timescale.db.models.fields
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("state", "0002_auto_20210915_2242"),
    ]

    operations = [
        migrations.CreateModel(
            name="Monitoring",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                (
                    "timestamp",
                    timescale.db.models.fields.TimescaleDateTimeField(
                        interval="1 day"
                    ),
                ),
                ("camera_uuid", models.TextField()),
                ("organization", models.TextField()),
                ("location", models.TextField()),
                ("zone", models.TextField()),
                ("camera_name", models.TextField()),
                (
                    "last_frame_processed_timestamp",
                    models.DateTimeField(null=True),
                ),
                (
                    "number_of_frames_waiting_to_be_processed",
                    models.IntegerField(null=True),
                ),
                ("ray_actor_id", models.TextField(null=True)),
            ],
        ),
    ]
