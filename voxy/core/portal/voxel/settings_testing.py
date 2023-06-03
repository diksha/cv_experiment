from core.portal.voxel.settings_helpers import AppConfig

app_config = AppConfig()

BASE_URL = app_config.get("BASE_URL", default="https//localhost:9000")

# ****************************************************************************
# Database
# ****************************************************************************
DATABASES = {
    # Run tests on hermetic SQLite databases
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
    },
    "state": {
        "ENGINE": "django.db.backends.sqlite3",
    },
}

# ****************************************************************************
# Cache
# ****************************************************************************
CACHES = {
    "default": {
        "BACKEND": "django_redis.cache.RedisCache",
        "LOCATION": "redis://dummy-cache",
        "OPTIONS": {
            "CLIENT_CLASS": "core.portal.voxel.cache_backends.CustomFakeStrictRedis",
        },
    }
}

# ****************************************************************************
# Security
# ****************************************************************************
SECRET_KEY = app_config.get("DJANGO_SECRET_KEY", default="__dummy_value__")
AWS_SIGNING_ROLE_ARN = "arn:aws:iam::123456789012:role/MyRole"
AWS_MULTI_REGION_ACCESS_ARN = (
    "arn:aws:s3::123456789012:accesspoint/testing.mrap"
)
