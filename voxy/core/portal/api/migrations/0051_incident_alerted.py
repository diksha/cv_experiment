# Generated by Django 4.0.3 on 2022-07-29 21:35

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0050_alter_comment_activity_type_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='incident',
            name='alerted',
            field=models.BooleanField(default=False, help_text='True if any alerts have been sent for this incident, otherwise False'),
        ),
    ]