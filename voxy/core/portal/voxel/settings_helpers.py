import json
import os
from functools import cached_property
from typing import Dict, List, Optional, Tuple, Union

from loguru import logger

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class AppConfig:
    @property
    def secrets_filepath(self) -> Optional[str]:
        """Secrets filepath.

        Returns:
            Optional[str]: filepath
        """
        return os.getenv("SECRETS_FILEPATH")

    @property
    def google_service_account_json_filepath(self) -> Optional[str]:
        """Google service account JSON filepath.

        Returns:
            Optional[str]: filepath
        """
        return os.getenv("SECRETS_GOOGLE_SERVICE_ACCOUNT_JSON_FILEPATH")

    @cached_property
    def google_service_account_json_string(self) -> Optional[str]:
        """Google service account JSON string property.

        Returns:
            Optional[str]: service account JSON string
        """
        try:
            with open(
                self.google_service_account_json_filepath,
                "r",
                encoding="utf-8",
            ) as file:
                return file.read()
        except (FileNotFoundError, TypeError):
            logger.info(
                "Google service account JSON not found at path:"
                f" {self.google_service_account_json_filepath}"
            )
            return None

    @cached_property
    def secrets_map(self) -> Dict[str, str]:
        """Return a dictionary of app config key/value pairs from secrets file.

        Returns:
            Dict[str, str]: dict of secrets
        """
        try:
            with open(self.secrets_filepath, "r", encoding="utf-8") as file:
                data = json.load(file)
        except (FileNotFoundError, TypeError):
            logger.info(
                f"Secrets file not found at path: {self.secrets_filepath}"
            )
            data = {}

        if self.google_service_account_json_string:
            data[
                "GOOGLE_SERVICE_ACCOUNT_JSON"
            ] = self.google_service_account_json_string

        return data

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get an application config value by key.

        Args:
            key (str): config key
            default (Optional[str], optional): default config value

        Returns:
            Optional[str]: config value
        """
        if key in self.secrets_map:
            logger.info(f"{key} read from secrets file")
            return self.secrets_map[key]

        if os.getenv(key):
            logger.info(f"{key} read from environment variable")
            return os.getenv(key)

        if default:
            logger.info(f"{key} read from default value")
            return default

        logger.info(f"{key} not found and no default provided")
        return None


# Singleton instance which can be used by settings_*.py files
app_config = AppConfig()


def get_templates_setting(
    loaders: List[Union[str, Tuple[str, List[str]]]] = None
) -> List[dict]:
    loaders = loaders if loaders else []
    return [
        {
            "BACKEND": "django.template.backends.django.DjangoTemplates",
            "DIRS": [
                os.path.join(BASE_DIR, "visualizer", "templates"),
            ],
            "OPTIONS": {
                "loaders": loaders,
                "context_processors": [
                    "django.template.context_processors.debug",
                    "django.template.context_processors.request",
                    "django.contrib.auth.context_processors.auth",
                    "django.contrib.messages.context_processors.messages",
                ],
            },
        },
    ]


def get_allowed_hosts(*base_list: Optional[List[str]]) -> List[str]:
    """Generates ALLOWED_HOSTS list, including dynamically provided hosts.

    Args:
        base_list (Optional[List[str]]): optional list of allowed host strings

    Returns:
        List[str]: final list of allowed host strings for the current environment
    """
    allowed_hosts = [
        *base_list,
        app_config.get("POD_IP"),
        app_config.get("HOST_IP"),
    ]
    valid_allowed_hosts = [h for h in allowed_hosts if h]
    return valid_allowed_hosts
