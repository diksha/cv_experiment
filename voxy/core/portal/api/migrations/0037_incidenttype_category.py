# Generated by Django 3.2.4 on 2022-02-24 03:22

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0036_auto_20220223_2147'),
    ]

    operations = [
        migrations.AddField(
            model_name='incidenttype',
            name='category',
            field=models.CharField(blank=True, choices=[('VEHICLE', 'VEHICLE'), ('ENVIRONMENT', 'ENVIRONMENT'), ('PEOPLE', 'PEOPLE')], max_length=25, null=True),
        ),
    ]
