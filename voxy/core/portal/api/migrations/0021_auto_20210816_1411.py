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

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("api", "0020_auto_20210720_1509"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="incident",
            name="incident_type",
        ),
        migrations.RenameField(
            model_name="incident",
            old_name="new_incident_type",
            new_name="incident_type",
        ),
    ]
