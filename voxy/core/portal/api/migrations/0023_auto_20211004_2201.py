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
        ("api", "0022_auto_20210908_2017"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="incident",
            name="video_thumbnail_url",
        ),
        migrations.RemoveField(
            model_name="incident",
            name="video_url",
        ),
        migrations.AlterField(
            model_name="incidenttype",
            name="organizations",
            field=models.ManyToManyField(
                related_name="incident_types",
                through="api.OrganizationIncidentType",
                to="api.Organization",
            ),
        ),
        migrations.AlterField(
            model_name="list",
            name="incidents",
            field=models.ManyToManyField(
                blank=True, related_name="lists", to="api.Incident"
            ),
        ),
    ]
