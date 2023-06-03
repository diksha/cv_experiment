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
        ("api", "0011_auto_20210417_1857"),
    ]

    operations = [
        migrations.AlterField(
            model_name="incident",
            name="incident_type",
            field=models.CharField(
                blank=True,
                choices=[
                    ("OPEN_DOOR_DURATION", "Open door duration"),
                    ("NO_STOP_AT_INTERSECTION", "No stop at intersection"),
                    ("PIGGYBACK", "Piggybacking"),
                    ("PARKING_DURATION", "Parking violation"),
                    ("DOOR_VIOLATION", "Door violation"),
                ],
                max_length=50,
                null=True,
            ),
        ),
    ]
