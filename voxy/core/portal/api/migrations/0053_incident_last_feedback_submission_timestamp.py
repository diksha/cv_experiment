# Generated by Django 4.1.2 on 2022-11-07 23:07

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0052_invitation_redeemed_invitation_role_invitation_zones'),
    ]

    operations = [
        migrations.AddField(
            model_name='incident',
            name='last_feedback_submission_timestamp',
            field=models.DateTimeField(blank=True, null=True),
        ),
    ]
