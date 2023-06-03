# Generated by Django 4.1.2 on 2022-11-27 05:41

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('state', '0018_state_state_state_end_tim_ed7f0a_idx'),
    ]

    operations = [
        migrations.RemoveIndex(
            model_name='state',
            name='state_state_end_tim_ed7f0a_idx',
        ),
        migrations.AddIndex(
            model_name='state',
            index=models.Index(fields=['timestamp', '-end_timestamp'], name='timestamp_range_idx'),
        ),
    ]
