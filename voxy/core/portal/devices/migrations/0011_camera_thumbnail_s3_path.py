# Generated by Django 4.0.3 on 2022-10-18 19:15

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('devices', '0010_alter_edge_lifecycle'),
    ]

    operations = [
        migrations.AddField(
            model_name='camera',
            name='thumbnail_s3_path',
            field=models.CharField(blank=True, max_length=250, null=True),
        ),
    ]
