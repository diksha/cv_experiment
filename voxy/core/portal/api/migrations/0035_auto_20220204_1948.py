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

import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ("api", "0034_incident_uuid"),
    ]

    operations = [
        migrations.AlterField(
            model_name="comment",
            name="incident",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name="comments",
                to="api.incident",
            ),
        ),
        migrations.AlterField(
            model_name="comment",
            name="owner",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                to=settings.AUTH_USER_MODEL,
            ),
        ),
    ]
