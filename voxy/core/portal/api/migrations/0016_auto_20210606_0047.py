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
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("api", "0015_list"),
    ]

    operations = [
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
                    (
                        "manage_incident_review_process",
                        "Can manage the incident review process.",
                    ),
                )
            },
        ),
        migrations.AddField(
            model_name="profile",
            name="organization",
            field=models.ForeignKey(
                null=True,
                on_delete=django.db.models.deletion.CASCADE,
                to="api.organization",
            ),
        ),
    ]
