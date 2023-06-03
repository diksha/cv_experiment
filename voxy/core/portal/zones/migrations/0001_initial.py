# Generated by Django 4.0.3 on 2022-04-21 18:53

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('api', '0039_delete_usersetting_incident_camera'),
    ]

    operations = [
        migrations.CreateModel(
            name='Zone',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('deleted_at', models.DateTimeField(blank=True, null=True)),
                ('name', models.CharField(max_length=250)),
                ('zone_type', models.CharField(choices=[('site', 'Site'), ('room', 'Room'), ('area', 'Area')], default='site', max_length=10)),
                ('organization', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='zones', to='api.organization')),
                ('parent_zone', models.ForeignKey(blank=True, max_length=250, null=True, on_delete=django.db.models.deletion.CASCADE, to='zones.zone')),
            ],
            options={
                'db_table': 'zones',
            },
        ),
    ]
