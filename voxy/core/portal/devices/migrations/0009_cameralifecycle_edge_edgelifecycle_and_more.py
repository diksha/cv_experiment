# Generated by Django 4.0.3 on 2022-10-12 21:35

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0052_invitation_redeemed_invitation_role_invitation_zones'),
        ('devices', '0008_camera_thumbnail_gcs_path'),
    ]

    operations = [
        migrations.CreateModel(
            name='CameraLifecycle',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('deleted_at', models.DateTimeField(blank=True, null=True)),
                ('key', models.CharField(help_text='\n        A key denoting the name of the camera lifecycle.\n        ', max_length=64, unique=True)),
                ('description', models.CharField(help_text='A description of the associated camera lifecycle.', max_length=256)),
            ],
            options={
                'db_table': 'camera_lifecycle',
            },
        ),
        migrations.CreateModel(
            name='Edge',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('uuid', models.UUIDField(auto_created=True, help_text='Unique identifier for the edge device.', unique=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('deleted_at', models.DateTimeField(blank=True, null=True)),
                ('mac_address', models.CharField(blank=True, help_text='MAC address of the edge device.', max_length=256, null=True)),
                ('name', models.CharField(help_text='User friendly name displayed throughout apps.', max_length=256)),
                ('serial', models.CharField(blank=True, help_text='Serial number of the edge device.', max_length=64, null=True)),
            ],
            options={
                'db_table': 'edge',
            },
        ),
        migrations.CreateModel(
            name='EdgeLifecycle',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('deleted_at', models.DateTimeField(blank=True, null=True)),
                ('key', models.CharField(help_text='\n        A key denoting the name of the edge lifecycle.\n        ', max_length=64, unique=True)),
                ('description', models.CharField(help_text='A description of the associated edge lifecycle.', max_length=256)),
            ],
            options={
                'db_table': 'edge_lifecycle',
            },
        ),
        migrations.AddIndex(
            model_name='camera',
            index=models.Index(fields=['organization'], name='camera_organiz_6af57f_idx'),
        ),
        migrations.AddIndex(
            model_name='camera',
            index=models.Index(fields=['zone'], name='camera_zone_id_b8610a_idx'),
        ),
        migrations.AddField(
            model_name='edge',
            name='lifecycle',
            field=models.ForeignKey(db_column='lifecycle', default='provisioning', on_delete=django.db.models.deletion.RESTRICT, to='devices.edgelifecycle', to_field='key'),
        ),
        migrations.AddField(
            model_name='edge',
            name='organization',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='edges', to='api.organization'),
        ),
        migrations.AddField(
            model_name='camera',
            name='edge',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.RESTRICT, to='devices.edge'),
        ),
        migrations.AddField(
            model_name='camera',
            name='lifecycle',
            field=models.ForeignKey(blank=True, db_column='lifecycle', null=True, on_delete=django.db.models.deletion.RESTRICT, to='devices.cameralifecycle', to_field='key'),
        ),
        migrations.AddIndex(
            model_name='edge',
            index=models.Index(fields=['organization'], name='edge_organiz_c9ab5b_idx'),
        ),
    ]