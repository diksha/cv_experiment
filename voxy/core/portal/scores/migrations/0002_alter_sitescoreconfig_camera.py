# Generated by Django 4.1.7 on 2023-02-27 03:35

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('devices', '0011_camera_thumbnail_s3_path'),
        ('scores', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='sitescoreconfig',
            name='camera',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to='devices.camera'),
        ),
    ]