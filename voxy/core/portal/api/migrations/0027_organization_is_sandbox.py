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
        ("api", "0026_auto_20211112_2134"),
    ]

    operations = [
        migrations.AddField(
            model_name="organization",
            name="is_sandbox",
            field=models.BooleanField(default=False),
        ),
    ]
