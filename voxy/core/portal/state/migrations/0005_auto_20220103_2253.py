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

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("state", "0004_auto_20211113_0508"),
    ]

    operations = [
        migrations.AlterField(
            model_name="event",
            name="event_type",
            field=models.IntegerField(
                choices=[
                    (0, "Unknown"),
                    (1, "Door Opened"),
                    (2, "Door Closed"),
                    (3, "Pit Entering Door"),
                    (4, "Pit Exiting Door"),
                    (5, "Door Partially Opened"),
                    (6, "Pit Entering Intersection"),
                    (7, "Pit Exiting Intersection"),
                    (8, "Pit Entering Aisle"),
                    (9, "Pit Exiting Aisle"),
                ]
            ),
        ),
        migrations.AlterField(
            model_name="state",
            name="actor_category",
            field=models.IntegerField(
                choices=[
                    (0, "Unknown"),
                    (1, "Person"),
                    (2, "Pit"),
                    (3, "Door"),
                    (4, "Hard Hat"),
                    (5, "Safety Vest"),
                    (6, "Bare Chest"),
                    (7, "Bare Head"),
                    (8, "Intersection"),
                    (9, "Aisle End"),
                ]
            ),
        ),
        migrations.AlterField(
            model_name="state",
            name="person_activity_type",
            field=models.IntegerField(
                choices=[(0, "Unknown"), (1, "Lifting"), (2, "Reaching")],
                null=True,
            ),
        ),
    ]
