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

import dj_database_url

from core.portal.voxel.settings_helpers import (
    app_config,
    get_templates_setting,
)

BASE_URL = "http://localhost:9000"
BASE_URL = app_config.get("BASE_URL", default="http://localhost:9000")

# ****************************************************************************
# Database
# ****************************************************************************
DEFAULT_DB_PORT = app_config.get("DEFAULT_DB_PORT", default=31003)
DATABASES = {
    "default": dj_database_url.parse(
        f"postgres://voxelapp:voxelvoxel@localhost:{DEFAULT_DB_PORT}/voxeldev",
        conn_max_age=600,
        ssl_require=False,
    ),
    "state": {
        "USER": "voxelapp",
        "PASSWORD": "voxelvoxel",
        "HOST": "localhost",
        "PORT": "31004",
        "ENGINE": "timescale.db.backends.postgresql",
        "NAME": "voxeldev",
    },
}

# ****************************************************************************
# Cache
# ****************************************************************************
CACHES = {
    "default": {
        "BACKEND": "django_redis.cache.RedisCache",
        "LOCATION": "redis://localhost:31005/0",
        "OPTIONS": {
            "CLIENT_CLASS": "django_redis.client.DefaultClient",
        },
    }
}

# ****************************************************************************
# Pub/Sub
# ****************************************************************************
PUBSUB_EMULATOR_HOST = "127.0.0.1:31002"
PUB_SUB_STATE_TOPIC = (
    "projects/local-project/topics/voxel-local-state-messages"
)
PUB_SUB_EVENT_TOPIC = (
    "projects/local-project/topics/voxel-local-event-messages"
)

# ****************************************************************************
# Auth
# ****************************************************************************
AUTH_PASSWORD_VALIDATORS = []

# ****************************************************************************
# Static files & templates
# ****************************************************************************
TEMPLATES = get_templates_setting(
    loaders=[
        "django.template.loaders.filesystem.Loader",
        "django.template.loaders.app_directories.Loader",
    ],
)

# ****************************************************************************
# Security
# ****************************************************************************
SECRET_KEY = app_config.get(
    "DJANGO_SECRET_KEY",
    default="pjx83elehve=b0%+d94kf*twj#fw*k487s-+67xs*7ia#tg4x3",
)
ALLOWED_HOSTS = ["localhost", ".ngrok.io", "portal-svc"]
CORS_ALLOWED_ORIGINS = [
    "http://localhost:9000",
    "http://localhost:9001",
]
CSRF_TRUSTED_ORIGINS = [
    "http://localhost:9000",
    "http://localhost:9001",
]

# ****************************************************************************
# Debugging
# ****************************************************************************
if app_config.get("DEBUG_SQL"):
    LOGGING = {
        "version": 1,
        "filters": {
            "require_debug_true": {
                "()": "django.utils.log.RequireDebugTrue",
            }
        },
        "handlers": {
            "console": {
                "level": "DEBUG",
                "filters": ["require_debug_true"],
                "class": "logging.StreamHandler",
            }
        },
        "loggers": {
            "django.db.backends": {
                "level": "DEBUG",
                "handlers": ["console"],
            }
        },
    }
