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

from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ("api", "0025_auto_20211112_0055"),
    ]

    operations = [
        migrations.AddField(
            model_name="incident",
            name="assigned_by",
            field=models.ManyToManyField(
                related_name="_incidents_assigned_by_me",
                through="api.UserIncident",
                to=settings.AUTH_USER_MODEL,
            ),
        ),
        migrations.AddField(
            model_name="incident",
            name="assigned_to",
            field=models.ManyToManyField(
                related_name="_incidents_assigned_to_me",
                through="api.UserIncident",
                to=settings.AUTH_USER_MODEL,
            ),
        ),
    ]
