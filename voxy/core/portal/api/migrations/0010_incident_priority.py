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
        ("api", "0009_auto_20210402_0237"),
    ]

    operations = [
        migrations.AddField(
            model_name="incident",
            name="priority",
            field=models.CharField(
                choices=[
                    ("low", "Low Priority"),
                    ("medium", "Medium Priority"),
                    ("high", "High Priority"),
                ],
                default="medium",
                max_length=20,
            ),
        ),
    ]