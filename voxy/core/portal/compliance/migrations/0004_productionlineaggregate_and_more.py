# Generated by Django 4.1.2 on 2022-11-15 17:21

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0053_incident_last_feedback_submission_timestamp'),
        ('zones', '0007_alter_zoneuser_user_alter_zoneuser_zone'),
        ('devices', '0011_camera_thumbnail_s3_path'),
        ('compliance', '0003_productionline'),
    ]

    operations = [
        migrations.CreateModel(
            name='ProductionLineAggregate',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('deleted_at', models.DateTimeField(blank=True, null=True)),
                ('group_key', models.DateTimeField(help_text='Timestamp of the beginning of the aggregate group.')),
                ('group_by', models.CharField(choices=[('HOUR', 'Hour'), ('DAY', 'Day'), ('MONTH', 'Month'), ('YEAR', 'Year')], help_text='How this data was grouped (by hour, by day, etc.)', max_length=25)),
                ('max_timestamp', models.DateTimeField(help_text='Max timestamp of all events contained in this aggregate group.')),
                ('uptime_duration_s', models.PositiveIntegerField(default=0)),
                ('downtime_duration_s', models.PositiveIntegerField(default=0)),
                ('camera', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='production_line_aggregates', to='devices.camera')),
                ('organization', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='production_line_aggregates', to='api.organization')),
                ('production_line', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='production_line_aggregates', to='compliance.productionline')),
                ('zone', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='production_line_aggregates', to='zones.zone')),
            ],
            options={
                'db_table': 'production_line_aggregate',
            },
        ),
        migrations.AddIndex(
            model_name='productionlineaggregate',
            index=models.Index(fields=['-group_key', 'group_by', 'organization', 'zone'], name='production__group_k_753af0_idx'),
        ),
        migrations.AddIndex(
            model_name='productionlineaggregate',
            index=models.Index(fields=['-group_key', 'group_by', 'production_line'], name='production__group_k_d952d7_idx'),
        ),
        migrations.AddConstraint(
            model_name='productionlineaggregate',
            constraint=models.UniqueConstraint(fields=('group_key', 'group_by', 'organization', 'zone', 'camera', 'production_line'), name='ergonomics_aggregate_unique_row_per_production_line_per_group_by_option'),
        ),
    ]