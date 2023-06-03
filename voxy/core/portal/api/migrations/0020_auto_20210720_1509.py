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
import itertools

import django.db.models.deletion
from django.db import migrations, models

EXISTING_TYPES = [
    {"value": "OPEN_DOOR_DURATION", "background_color": "#ecd361"},
    {"value": "NO_STOP_AT_INTERSECTION", "background_color": "#8a291d"},
    {"value": "PIGGYBACK", "background_color": "#734c83"},
    {"value": "PARKING_DURATION", "background_color": "#2980b9"},
    {"value": "DOOR_VIOLATION", "background_color": "#16a085"},
    {"value": "MISSING_PPE", "background_color": "#dd7700"},
    {"value": "BAD_POSTURE", "background_color": "#ff9900"},
]


def forward_insert_types(apps, schema_editor):
    IncidentType = apps.get_model("api", "IncidentType")
    Organization = apps.get_model("api", "Organization")
    OrganizationIncidentType = apps.get_model(
        "api", "OrganizationIncidentType"
    )

    db_alias = schema_editor.connection.alias

    # Apply incident type for all companies for now
    list_organizations = Organization.objects.all()
    objs = IncidentType.objects.using(db_alias).bulk_create(
        [IncidentType(**val) for val in EXISTING_TYPES]
    )
    pairs = itertools.product(list_organizations, objs)
    OrganizationIncidentType.objects.using(db_alias).bulk_create(
        [
            OrganizationIncidentType(
                organization=pair[0], incident_type=pair[1]
            )
            for pair in pairs
        ]
    )


def forward_update_incident(apps, schema_editor):
    Incident = apps.get_model("api", "Incident")
    IncidentType = apps.get_model("api", "IncidentType")

    db_alias = schema_editor.connection.alias
    new_type_map = {val.value: val for val in IncidentType.objects.all()}

    for obj in Incident.objects.using(db_alias).all():
        obj.new_incident_type = new_type_map.get(obj.incident_type)
        obj.save()


def reverse_update_incident(apps, schema_editor):
    Incident = apps.get_model("api", "Incident")
    db_alias = schema_editor.connection.alias
    Incident.objects.using(db_alias).filter(
        new_incident_type__isnull=False
    ).update(new_incident_type=None)


class Migration(migrations.Migration):

    dependencies = [
        ("api", "0019_auto_20210703_1948"),
    ]

    operations = [
        migrations.CreateModel(
            name="IncidentType",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                ("deleted_at", models.DateTimeField(blank=True, null=True)),
                ("value", models.CharField(max_length=100, unique=True)),
                ("background_color", models.CharField(max_length=7)),
            ],
            options={
                "abstract": False,
            },
        ),
        migrations.AlterField(
            model_name="incident",
            name="organization",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.CASCADE,
                to="api.organization",
            ),
        ),
        migrations.CreateModel(
            name="OrganizationIncidentType",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                ("deleted_at", models.DateTimeField(blank=True, null=True)),
                ("enabled", models.BooleanField(default=True)),
                (
                    "incident_type",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to="api.incidenttype",
                    ),
                ),
                (
                    "organization",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to="api.organization",
                    ),
                ),
            ],
            options={
                "unique_together": {("incident_type", "organization")},
            },
        ),
        migrations.AddField(
            model_name="incidenttype",
            name="organizations",
            field=models.ManyToManyField(
                through="api.OrganizationIncidentType", to="api.Organization"
            ),
        ),
        migrations.AddField(
            model_name="incident",
            name="new_incident_type",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.CASCADE,
                to="api.incidenttype",
            ),
        ),
        migrations.RunPython(forward_insert_types, migrations.RunPython.noop),
        migrations.RunPython(forward_update_incident, reverse_update_incident),
    ]
