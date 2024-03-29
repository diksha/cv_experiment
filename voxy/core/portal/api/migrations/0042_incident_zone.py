# Generated by Django 4.0.3 on 2022-05-02 17:00

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('zones', '0003_alter_zone_organization_and_more'),
        ('api', '0041_alter_incident_camera'),
    ]

    operations = [
        migrations.AddField(
            model_name='incident',
            name='zone',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.RESTRICT, related_name='incidents', to='zones.zone'),
        ),
    ]
