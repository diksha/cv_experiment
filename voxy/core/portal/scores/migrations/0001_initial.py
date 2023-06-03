# Generated by Django 4.1.7 on 2023-02-22 20:37

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('zones', '0011_zone_config'),
        ('devices', '0011_camera_thumbnail_s3_path'),
        ('perceived_data', '0009_remove_perceivedactorstatedurationaggregate_perceived_actor_state_duration_aggregate_unique_constrai'),
    ]

    operations = [
        migrations.CreateModel(
            name='ScoreBand',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('deleted_at', models.DateTimeField(blank=True, null=True)),
                ('name', models.CharField(max_length=100, unique=True)),
            ],
            options={
                'db_table': 'score_band',
            },
        ),
        migrations.CreateModel(
            name='ScoreDefinition',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('deleted_at', models.DateTimeField(blank=True, null=True)),
                ('name', models.CharField(max_length=100)),
                ('calculation_method', models.PositiveSmallIntegerField(choices=[(0, 'Not Implemented'), (1, 'Thirty Day Event Score')], default=0)),
                ('perceived_event_rate_definition', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='perceived_data.perceivedeventratedefinition')),
                ('score_band', models.ForeignKey(on_delete=django.db.models.deletion.PROTECT, to='scores.scoreband')),
            ],
            options={
                'db_table': 'score_definition',
            },
        ),
        migrations.CreateModel(
            name='SiteScoreConfig',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('deleted_at', models.DateTimeField(blank=True, null=True)),
                ('name', models.CharField(max_length=100, unique=True)),
                ('camera', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='devices.camera')),
                ('score_definition', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='scores.scoredefinition')),
                ('site', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='zones.zone')),
            ],
            options={
                'db_table': 'site_score_config',
            },
        ),
        migrations.CreateModel(
            name='ScoreBandRange',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('deleted_at', models.DateTimeField(blank=True, null=True)),
                ('lower_bound_inclusive', models.DecimalField(decimal_places=10, default=0.0, help_text='The lower bound of the range', max_digits=19)),
                ('score_value', models.PositiveSmallIntegerField()),
                ('score_band', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='scores.scoreband')),
            ],
            options={
                'db_table': 'score_band_range',
            },
        ),
        migrations.AddConstraint(
            model_name='sitescoreconfig',
            constraint=models.UniqueConstraint(fields=('site', 'camera', 'score_definition'), name='site_score_config_unique_constraint'),
        ),
        migrations.AddConstraint(
            model_name='scoredefinition',
            constraint=models.UniqueConstraint(fields=('perceived_event_rate_definition', 'score_band', 'calculation_method'), name='score_definition_unique_constraint'),
        ),
        migrations.AddConstraint(
            model_name='scorebandrange',
            constraint=models.UniqueConstraint(fields=('score_band', 'lower_bound_inclusive'), name='score_band_range_unique_constraint'),
        ),
    ]