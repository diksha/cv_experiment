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
        ("api", "0012_auto_20210425_1801"),
    ]

    operations = [
        migrations.AlterModelOptions(
            name="incident",
            options={},
        ),
        migrations.AlterModelOptions(
            name="profile",
            options={
                "permissions": (
                    ("can_access_incident_feed", "Can access incident feed."),
                    (
                        "can_access_analytics_page",
                        "Can access analytics page.",
                    ),
                    ("can_access_live_feed", "Can access live feed."),
                    ("can_review_incidents", "Can review incidents."),
                )
            },
        ),
    ]
