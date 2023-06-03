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
        ("state", "0005_auto_20220103_2253"),
    ]

    operations = [
        migrations.AddField(
            model_name="state",
            name="pit_is_stationary",
            field=models.BooleanField(null=True),
        ),
    ]
