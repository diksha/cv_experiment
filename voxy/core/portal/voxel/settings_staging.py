#
# Copyright 2020-2021 Voxel Labs, Inc.
# All rights reserved.
#
# This document may not be reproduced, republished, distributed, transmitted,
# displayed, broadcast or otherwise exploited in any manner without the express
# prior written permission of Voxel Labs, Inc. The receipt or possession of this
# document does not convey any rights to reproduce, disclose, or distribute its
# contents, or to manufacture, use, or sell anything that it may describe, in
# whole or in part.
#
import os

import sentry_sdk
from sentry_sdk.integrations.django import DjangoIntegration

from core.portal.voxel.settings_helpers import (
    BASE_DIR,
    app_config,
    get_allowed_hosts,
    get_templates_setting,
)

BASE_URL = app_config.get(
    "BASE_URL", default="https://app.staging.voxelai.com"
)

DEPLOYMENT_PROVIDER = app_config.get(
    "DEPLOYMENT_PROVIDER",
    default="ecs",
)

AWS_SIGNING_ROLE_ARN = "arn:aws:iam::115099983231:role/s3-signing-role"
AWS_MULTI_REGION_ACCESS_ARN = (
    "arn:aws:s3::115099983231:accesspoint/men1h3wgx7wyh.mrap"
)

# ****************************************************************************
# Sentry
# ****************************************************************************
sentry_sdk.init(
    dsn="https://2705321fb58548b48c8ef6f0d6def646@sentry.private.voxelplatform.com/7",
    integrations=[DjangoIntegration()],
    # Set traces_sample_rate to 1.0 to capture 100%
    # of transactions for performance monitoring.
    # We recommend adjusting this value in production.
    traces_sample_rate=0.25,
    # If you wish to associate users to errors (assuming you are using
    # django.contrib.auth) you may enable sending PII data.
    send_default_pii=True,
)

# ****************************************************************************
# Cache
# ****************************************************************************
CACHES = {
    "default": {
        "BACKEND": "django_redis.cache.RedisCache",
        "LOCATION": app_config.get("REDIS_LOCATION"),
        "OPTIONS": {
            "CLIENT_CLASS": "django_redis.client.DefaultClient",
        },
    }
}

# ****************************************************************************
# Pub/Sub
# ****************************************************************************
PUB_SUB_STATE_TOPIC = (
    "projects/sodium-carving-227300/topics/voxel-staging-state-messages"
)
PUB_SUB_EVENT_TOPIC = (
    "projects/sodium-carving-227300/topics/voxel-staging-event-messages"
)

# ****************************************************************************
# Static files & templates
# ****************************************************************************
TEMPLATES = get_templates_setting(
    loaders=[
        "django.template.loaders.filesystem.Loader",
        "django.template.loaders.app_directories.Loader",
        "voxel.template_loaders.S3StorageLoader",
    ],
)
STATICFILES_DIRS = [
    # Used by collectstatic to move frontend build artifacts to /static
    ("frontend", "services/portal/web/build"),
    os.path.join(BASE_DIR, "visualizer/templates"),
]
DEFAULT_FILE_STORAGE = "storages.backends.s3boto3.S3Boto3Storage"
STATICFILES_STORAGE = "storages.backends.s3boto3.S3StaticStorage"
# TODO: Remove DEPLOYMENT_ENV environment variable once migrated
AWS_STORAGE_BUCKET_NAME = (
    "voxel-portal-staging-static-resources"
    if DEPLOYMENT_PROVIDER == "eks-final"
    else "voxel-portal-staging-static"
)
AWS_LOCATION = "static/"
AWS_S3_CUSTOM_DOMAIN = (
    "app.staging.voxelplatform.com"
    if DEPLOYMENT_PROVIDER == "eks-final"
    else "app.staging.voxelai.com"
)

# ****************************************************************************
# Security
# ****************************************************************************
SECRET_KEY = app_config.get("DJANGO_SECRET_KEY")
ALLOWED_HOSTS = get_allowed_hosts(
    "app.staging.voxelai.com",
    "app.staging.voxelplatform.com",
)
ALLOWED_CIDR_NETS = [
    # Allow internal VPC hosts for things like health checks
    "172.16.0.0/12",
]
CORS_ALLOWED_ORIGINS = [
    "https://app.staging.voxelai.com",
    "https://app.staging.voxelplatform.com",
]
CSRF_TRUSTED_ORIGINS = [
    "https://app.staging.voxelai.com",
    "https://app.staging.voxelplatform.com",
]

CSRF_COOKIE_SECURE = True
SESSION_COOKIE_SECURE = True
# ****************************************************************************
# Email
# ****************************************************************************
SEND_TRANSACTIONAL_EMAILS = True
