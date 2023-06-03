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
        ("api", "0006_auto_20210107_0544"),
    ]

    operations = [
        migrations.AddField(
            model_name="profile",
            name="data",
            field=models.JSONField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name="incident",
            name="data",
            field=models.JSONField(blank=True, null=True),
        ),
    ]
