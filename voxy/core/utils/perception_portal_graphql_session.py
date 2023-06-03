import json
from ast import literal_eval
from datetime import datetime, timedelta

import requests
from botocore.exceptions import ClientError
from loguru import logger

from core.utils.aws_utils import (
    get_secret_from_aws_secret_manager,
    update_aws_secret,
)


class PerceptionPortalSession:

    # trunk-ignore(bandit/B105)
    AUTH_TOKEN_SECRET_NAME = "PERCEPTION_PORTAL_AUTH_TOKEN"
    GEN_TIME_KEY = "gen_time"
    # trunk-ignore(bandit/B105)
    ACCESS_TOKEN_KEY = "access_token"

    def __init__(
        self, environment, cache_file_name="perception_portal_auth_token"
    ):
        self.environment = environment
        self.session = requests.Session()
        self.credentials = literal_eval(
            get_secret_from_aws_secret_manager(
                f"{self.environment}_PERCEPTION_PORTAL_AUTH",
            )
        )
        self.host = self.credentials["host"]

        access_token = self._generate_auth_token()
        self.headers = {"authorization": f"Bearer {access_token}"}

    def _generate_auth_token(self) -> str:
        """Fetch an auth0 token. If the token is already cached, it fetches that else
        a new token is generated.

        [IMPORTANT] Note that it is quite cruicial to cache and use m2m tokens.
        It is not only standard practice but is especially important for us to do
        because we are on a limited plan for auth0 m2m tokens. This means we have close
        to 1k tokens to use a month. The tokens also have a 24 hr validity (as of Jan 30 2021).
        Thus we cache tokens in aws secrets manager so that any we can restrict the number of
        tokens generated per day and use them more efficiently.

        Returns:
            str: auth0 access token
        """
        try:
            # first check if a valid token already exists by looking
            # for in AWS secrets manager
            token_metadata = literal_eval(
                get_secret_from_aws_secret_manager(
                    f"{self.AUTH_TOKEN_SECRET_NAME}",
                )
            )
            access_token = None
            gen_time = datetime.strptime(
                token_metadata[self.GEN_TIME_KEY], "%m/%d/%y %H:%M:%S"
            )
            gen_time_str = gen_time.strftime("%m/%d/%y %H:%M:%S")
            if datetime.now() - gen_time >= timedelta(days=1):
                # expired token, regenerate
                data = {
                    "client_id": self.credentials["client_id"],
                    "client_secret": self.credentials["client_secret"],
                    "audience": self.credentials["audience"],
                    "grant_type": "client_credentials",
                }
                response = self.session.post(
                    self.credentials["auth_url"], data=data
                )
                access_token = json.loads(response.text)["access_token"]
                # write/update secrets manager with access token
                token_metadata = {
                    self.GEN_TIME_KEY: datetime.now().strftime(
                        "%m/%d/%y %H:%M:%S"
                    ),
                    self.ACCESS_TOKEN_KEY: str(access_token),
                }

                update_aws_secret(
                    secret_id=f"{self.AUTH_TOKEN_SECRET_NAME}",
                    secret_value=json.dumps(token_metadata),
                )
                logger.info(f"Generated a new access token at {gen_time_str}.")
            else:
                access_token = token_metadata[self.ACCESS_TOKEN_KEY]
                logger.info(
                    f"Fetching cached token generated at {gen_time_str}"
                )
        except ClientError as exc:
            logger.warn(
                f"Failed to fetch auth token from secret manager {exc}"
            )

        return access_token

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.session.close()
