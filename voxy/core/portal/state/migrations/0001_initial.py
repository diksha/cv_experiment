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

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="Event",
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
                ("subject_id", models.CharField(max_length=64, null=True)),
                ("object_id", models.CharField(max_length=64, null=True)),
                (
                    "event_type",
                    models.IntegerField(
                        choices=[
                            (0, "Unknown"),
                            (1, "Door Opened"),
                            (2, "Door Closed"),
                            (3, "Pit Entering Door"),
                            (4, "Pit Exiting Door"),
                        ]
                    ),
                ),
                ("end_timestamp", models.DateTimeField()),
                ("run_uuid", models.CharField(max_length=64, null=True)),
            ],
        ),
        migrations.CreateModel(
            name="State",
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
                ("actor_id", models.CharField(max_length=64)),
                (
                    "actor_category",
                    models.IntegerField(
                        choices=[
                            (0, "Unknown"),
                            (1, "Person"),
                            (2, "Pit"),
                            (3, "Door"),
                            (4, "Hard Hat"),
                            (5, "Safety Vest"),
                        ]
                    ),
                ),
                ("end_timestamp", models.DateTimeField()),
                ("run_uuid", models.CharField(max_length=64, null=True)),
                ("door_is_open", models.BooleanField(null=True)),
            ],
        ),
    ]
