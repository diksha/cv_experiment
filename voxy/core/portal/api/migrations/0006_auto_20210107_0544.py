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
        ("api", "0005_organization"),
    ]

    operations = [
        migrations.AddField(
            model_name="incident",
            name="organization",
            field=models.ForeignKey(
                default=1,
                on_delete=django.db.models.deletion.CASCADE,
                to="api.organization",
            ),
            preserve_default=False,
        ),
        migrations.AlterField(
            model_name="organization",
            name="users",
            field=models.ManyToManyField(
                blank=True,
                related_name="organizations",
                to=settings.AUTH_USER_MODEL,
            ),
        ),
    ]
