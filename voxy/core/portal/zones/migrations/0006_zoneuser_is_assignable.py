# Generated by Django 4.0.3 on 2022-09-15 16:19

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('zones', '0005_zone_timezone'),
    ]

    operations = [
        migrations.AddField(
            model_name='zoneuser',
            name='is_assignable',
            field=models.BooleanField(default=True),
        ),
    ]
