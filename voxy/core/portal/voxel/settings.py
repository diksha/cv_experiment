import os
import sys
from typing import Dict, List

from corsheaders.defaults import default_headers
from dotenv import load_dotenv

from core.portal.voxel.settings_helpers import BASE_DIR, app_config

load_dotenv()

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
WSGI_APPLICATION = "voxel.wsgi.application"

ENVIRONMENT = app_config.get("ENVIRONMENT")
PRODUCTION = ENVIRONMENT == "production"
STAGING = ENVIRONMENT == "staging"
DEVELOPMENT = ENVIRONMENT == "development"
TEST = (
    ENVIRONMENT == "test" or "test" in sys.argv or "test_coverage" in sys.argv
)

AWS_SIGNING_ROLE_ARN = None
AWS_MULTI_REGION_ACCESS_ARN = None

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = None

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = DEVELOPMENT or app_config.get("DJANGO_DEBUG", default=False)

CORS_ALLOW_CREDENTIALS = True
CORS_ALLOW_HEADERS = list(default_headers) + [
    "x-voxel-client-timezone",
    "x-voxel-client-timezone-offset",
    "sentry-trace",
]

# Application definition

INSTALLED_APPS = [
    # Pre-Django apps
    "corsheaders",
    # Django apps
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "django.contrib.sites",
    "django_extensions",
    "django_filters",
    "django_plotly_dash.apps.DjangoPlotlyDashConfig",
    # Local apps
    "core.portal.accounts.apps.Config",
    "core.portal.activity.apps.Config",
    "core.portal.analytics.apps.Config",
    "core.portal.api.apps.Config",
    "core.portal.devices.apps.Config",
    "core.portal.comments.apps.Config",
    "core.portal.compliance.apps.Config",
    "core.portal.incident_feedback.apps.Config",
    "core.portal.incidents.apps.Config",
    "core.portal.motion_zones.apps.Config",
    "core.portal.notifications.apps.Config",
    "core.portal.organizations.apps.Config",
    "core.portal.perceived_data.apps.Config",
    "core.portal.scores.apps.Config",
    "core.portal.integrations.apps.Config",
    "core.portal.session.apps.Config",
    "core.portal.state.apps.Config",
    "core.portal.voxel.management",
    "core.portal.zones.apps.Config",
    # Post-Django apps
    "rest_framework",
    "rest_framework.authtoken",
    "graphene_django",
]

MIDDLEWARE = [
    "allow_cidr.middleware.AllowCIDRMiddleware",
    "corsheaders.middleware.CorsMiddleware",
    "core.portal.voxel.middleware.CustomHeaderMiddleware",
    "core.portal.voxel.middleware.CustomSecurityMiddleware",
    "django.middleware.security.SecurityMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "core.portal.accounts.middleware.StatelessAuthenticationMiddleware",
    "core.portal.accounts.middleware.PermissionsMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
    "django_plotly_dash.middleware.BaseMiddleware",
    "core.portal.activity.middleware.SessionTrackingMiddleware",
]

SITE_ID = 2  # Required for django-rest-auth, allauth

PLOTLY_DASH = {
    "view_decorator": "django_plotly_dash.access.login_required",
}

X_FRAME_OPTIONS = "SAMEORIGIN"

AUTHENTICATION_BACKENDS = (
    "core.portal.accounts.backends.Auth0Backend",
    # TODO(PRO-293): remove this entry after removing Django permission usage,
    #                user.has_perm() only works if this backend is present.
    "django.contrib.auth.backends.ModelBackend",
)

ROOT_URLCONF = "core.portal.voxel.urls"
APPEND_SLASH = False

# ****************************************************************************
# Static files & templates
# ****************************************************************************
STATIC_URL = "/static/"
STATIC_ROOT = os.path.join(BASE_DIR, "static")
TEMPLATES = []

STATICFILES_DIRS = [os.path.join(BASE_DIR, "visualizer/templates")]

REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": [
        "rest_framework.authentication.SessionAuthentication",
        "rest_framework.authentication.TokenAuthentication",
    ],
    "DEFAULT_PAGINATION_CLASS": "rest_framework.pagination.PageNumberPagination",  # noqa
    "PAGE_SIZE": 10,
    "DEFAULT_PERMISSION_CLASSES": [
        "core.portal.api.permissions.NoPermission",
    ],  # noqa
}

# ****************************************************************************
# Pub/Sub
# ****************************************************************************
PUBSUB_EMULATOR_HOST = None
PUB_SUB_STATE_TOPIC = None
PUB_SUB_EVENT_TOPIC = None

# ****************************************************************************
# Auth0
# ****************************************************************************
AUTH0_TENANT_DOMAIN = app_config.get("AUTH0_TENANT_DOMAIN")
AUTH0_CUSTOM_DOMAIN = app_config.get("AUTH0_CUSTOM_DOMAIN")
AUTH0_DOMAIN = AUTH0_CUSTOM_DOMAIN or AUTH0_TENANT_DOMAIN
AUTH0_JWKS_URI = f"https://{AUTH0_DOMAIN}/.well-known/jwks.json"
AUTH0_JWT_ISSUER = f"https://{AUTH0_DOMAIN}/"
AUTH0_API_IDENTIFIER = app_config.get("AUTH0_API_IDENTIFIER")
AUTH0_JWT_AUDIENCE = app_config.get("AUTH0_JWT_AUDIENCE")
AUTH0_MANAGEMENT_CLIENT_ID = app_config.get("AUTH0_MANAGEMENT_CLIENT_ID")
AUTH0_MANAGEMENT_CLIENT_SECRET = app_config.get(
    "AUTH0_MANAGEMENT_CLIENT_SECRET"
)

# ****************************************************************************
# Google Cloud Platform
# ****************************************************************************
GOOGLE_SERVICE_ACCOUNT_JSON = app_config.get("GOOGLE_SERVICE_ACCOUNT_JSON")

# ****************************************************************************
# GraphQL, graphene-django
# ****************************************************************************
GRAPHENE = {
    "SCHEMA": "core.portal.lib.graphql.graphene_schema.schema",
    "ATOMIC_MUTATIONS": True,
    "MIDDLEWARE": [
        "core.portal.lib.graphql.middleware.localization.LocalizationMiddleware",
    ],
}

# ****************************************************************************
# Password validation
# https://docs.djangoproject.com/en/2.2/ref/settings/#auth-password-validators
# ****************************************************************************
AUTH_PASSWORD_VALIDATORS: List[Dict[str, str]] = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator",  # noqa
    },
    {
        "NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",  # noqa
    },
    {
        "NAME": "django.contrib.auth.password_validation.CommonPasswordValidator",  # noqa
    },
    {
        "NAME": "django.contrib.auth.password_validation.NumericPasswordValidator",  # noqa
    },
]

# ****************************************************************************
# AllAuth
# ****************************************************************************
ACCOUNT_EMAIL_REQUIRED = True
ACCOUNT_AUTHENTICATION_METHOD = "email"
ACCOUNT_EMAIL_SUBJECT_PREFIX = ""
ACCOUNT_EMAIL_VERIFICATION = "none"

# ****************************************************************************
# Internationalization
# https://docs.djangoproject.com/en/2.2/topics/i18n/
# ****************************************************************************
LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

# ****************************************************************************
# Email sending
# ****************************************************************************
SEND_TRANSACTIONAL_EMAILS = app_config.get("SEND_TRANSACTIONAL_EMAILS")
SENDGRID_API_KEY = app_config.get("SENDGRID_API_KEY")
EMAIL_BACKEND = "django.core.mail.backends.smtp.EmailBackend"
EMAIL_HOST = "smtp.sendgrid.net"
EMAIL_PORT = 587
EMAIL_USE_TLS = True
EMAIL_HOST_USER = "apikey"
EMAIL_HOST_PASSWORD = SENDGRID_API_KEY
DEFAULT_FROM_EMAIL = "Voxel <no-reply@voxelai.com>"
CUSTOMER_SUMMARY_EMAILS_BCC = "customer-summary-emails-sandbox@voxelai.com"
CUSTOMER_PULSE_EMAILS_BCC = "customer-pulse-emails-sandbox@voxelai.com"
CUSTOMER_ALERT_EMAILS_BCC = "customer-alerts-sandbox@voxelai.com"

# ****************************************************************************
# Database
# ****************************************************************************
DATABASE_ROUTERS = ["core.portal.voxel.dbrouters.DbRouter"]
DATABASES = {
    "default": {
        "USER": app_config.get("DEFAULT_DATABASE_USER"),
        "PASSWORD": app_config.get("DEFAULT_DATABASE_PASSWORD"),
        "NAME": app_config.get("DEFAULT_DATABASE_NAME"),
        "PORT": app_config.get("DEFAULT_DATABASE_PORT", 5432),
        "HOST": app_config.get("DEFAULT_DATABASE_HOST"),
        "ENGINE": "django.db.backends.postgresql",
        "OPTIONS": {"connect_timeout": 30},
    },
    "state": {
        "USER": app_config.get("STATE_DATABASE_USER"),
        "PASSWORD": app_config.get("STATE_DATABASE_PASSWORD"),
        "NAME": app_config.get("STATE_DATABASE_NAME"),
        "HOST": app_config.get("STATE_DATABASE_HOST"),
        "PORT": app_config.get("STATE_DATABASE_PORT"),
        "ENGINE": "timescale.db.backends.postgresql",
        "OPTIONS": {
            "sslmode": "require",
        },
    },
}

# ****************************************************************************
# Logging
# ****************************************************************************
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
        },
    },
    "root": {
        "handlers": ["console"],
        "level": app_config.get("DJANGO_LOG_LEVEL", "INFO"),
    },
}

# ****************************************************************************
# Environment-specific settings
# ****************************************************************************
if PRODUCTION:
    # trunk-ignore(pylint/W0401,flake8/F401,flake8/F403)
    from core.portal.voxel.settings_production import *
elif STAGING:
    # trunk-ignore(pylint/W0401,flake8/F401,flake8/F403)
    from core.portal.voxel.settings_staging import *
elif DEVELOPMENT:
    # trunk-ignore(pylint/W0401,flake8/F401,flake8/F403)
    from core.portal.voxel.settings_development import *
elif TEST:
    # trunk-ignore(pylint/W0401,flake8/F401,flake8/F403,pylint/W0614)
    from core.portal.voxel.settings_testing import *
else:
    raise RuntimeError(
        "No environment specified. Please specify an environment with an env"
        " variable or secret using the ENVIRONMENT key."
    )
