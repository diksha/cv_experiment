import datetime
from typing import Optional

import boto3
from django.conf import settings

from core.utils import aws_utils

_TIMEOUT_MINUTES = 30
_TIMEOUT_SECONDS = _TIMEOUT_MINUTES * 60
_PADDING_TIME_SECONDS = 5 * 60


class SignedURLManager:
    """Class for interacting with cloud storage signed URLs."""

    _client = None
    _client_expiration_time = None

    def _get_boto3_client(self) -> boto3.Session.client:
        """Get boto3 client"""
        if settings.DEVELOPMENT:
            return None

        self.refresh_client_if_needed()
        return self._client

    def _create_boto3_client(self) -> boto3.Session.client:
        """Create a new boto3 client with the specified role."""
        sts_client = boto3.client("sts")

        assumed_role_object = sts_client.assume_role(
            RoleArn=settings.AWS_SIGNING_ROLE_ARN,
            RoleSessionName="AssumeRoleSession",
        )

        credentials = assumed_role_object["Credentials"]

        client = boto3.client("s3")
        self._client_expiration_time = credentials["Expiration"]

        return client

    def refresh_client_if_needed(self) -> None:
        """Refresh the boto3 client if the credentials are about to expire."""
        if (
            self._client is None
            or self._client_expiration_time is None
            or (
                self._client_expiration_time
                - datetime.datetime.now(datetime.timezone.utc)
            ).total_seconds()
            <= _PADDING_TIME_SECONDS
        ):
            self._client = self._create_boto3_client()

    def get_signed_url(
        self, s3_path: Optional[str] = None, enable_multi_region_access=False
    ) -> str:
        """Returns a signed URL for the provided path.

        If the value is reasonably fresh in the cache, the cached value is
        returned. Otherwise a new signed URL is generated and cached.

        Args:
            s3_path (str): The fully qualified S3 path (s3://...)
            enable_multi_region_access (bool): whether to enable multi-region access.

        Returns:
            str: A signed URL for the requested asset.
        """
        bucket, filepath = aws_utils.separate_bucket_from_relative_path(
            s3_path
        )

        if settings.AWS_MULTI_REGION_ACCESS_ARN and enable_multi_region_access:
            bucket = settings.AWS_MULTI_REGION_ACCESS_ARN

        signed_url = aws_utils.generate_presigned_url(
            bucket,
            filepath,
            timeout=_TIMEOUT_SECONDS,
            s3_client=self._get_boto3_client(),
        )

        return signed_url


signed_url_manager = SignedURLManager()
