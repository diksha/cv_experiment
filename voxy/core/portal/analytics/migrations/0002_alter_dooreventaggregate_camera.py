# Generated by Django 4.0.3 on 2022-10-12 20:26

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('devices', '0008_camera_thumbnail_gcs_path'),
        ('analytics', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='dooreventaggregate',
            name='camera',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='door_event_aggregates', to='devices.camera'),
        ),
    ]
