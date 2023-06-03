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


def populate_key_and_name_fields(apps, *_):
    IncidentType = apps.get_model("api", "IncidentType")

    for incident_type in IncidentType.objects.all():
        incident_type.key = incident_type.value
        incident_type.name = incident_type.value.replace("_", " ").title()
        incident_type.save()


class Migration(migrations.Migration):

    dependencies = [
        ("api", "0029_auto_20211127_1642"),
    ]

    operations = [
        migrations.AddField(
            model_name="incidenttype",
            name="key",
            field=models.CharField(
                blank=True, max_length=100, null=True, unique=True
            ),
        ),
        migrations.AddField(
            model_name="incidenttype",
            name="name",
            field=models.CharField(
                blank=True, max_length=100, null=True, unique=True
            ),
        ),
        migrations.RunPython(
            populate_key_and_name_fields, migrations.RunPython.noop
        ),
        migrations.AlterField(
            model_name="incidenttype",
            name="key",
            field=models.CharField(
                blank=False, max_length=100, null=False, unique=True
            ),
        ),
        migrations.AlterField(
            model_name="incidenttype",
            name="name",
            field=models.CharField(
                blank=False, max_length=100, null=False, unique=True
            ),
        ),
    ]
