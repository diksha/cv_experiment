# Generated by Django 4.1.7 on 2023-02-22 20:37

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0064_incident_cooldown_source_incident_review_status'),
    ]

    operations = [
        migrations.AlterField(
            model_name='incident',
            name='review_status',
            field=models.PositiveSmallIntegerField(choices=[(1, 'Needs Review'), (2, 'Do Not Review'), (3, 'Valid'), (4, 'Invalid'), (5, 'Valid And Needs Review')], null=True),
        ),
    ]