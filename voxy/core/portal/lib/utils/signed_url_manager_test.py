import datetime
from unittest.mock import MagicMock, patch

from django.conf import settings

from core.portal.lib.utils.signed_url_manager import (
    _PADDING_TIME_SECONDS,
    SignedURLManager,
)

# trunk-ignore-all(pylint/W0212): accessing priv methods for test


def create_mock_sts_client() -> MagicMock:
    """Create a mock STS client.

    Returns:
        MagicMock: Mocked STS client.
    """
    mock_sts_client = MagicMock()
    mock_sts_client.return_value.assume_role.return_value = {
        "Credentials": {
            "Expiration": datetime.datetime.now(datetime.timezone.utc)
            + datetime.timedelta(seconds=_PADDING_TIME_SECONDS + 100),
        }
    }
    return mock_sts_client


def create_mock_generate_presigned_url() -> MagicMock:
    """Create a mock function for generating presigned URLs.

    Returns:
        MagicMock: Mocked generate_presigned_url function.
    """
    mock_generate_presigned_url = MagicMock()
    mock_generate_presigned_url.return_value = (
        "https://example.com/presigned-url"
    )
    return mock_generate_presigned_url


def test_create_boto3_client() -> None:
    """Test for creating boto3 client."""
    mock_sts_client = create_mock_sts_client()
    mock_s3_client = MagicMock()

    with patch("boto3.client", side_effect=[mock_sts_client, mock_s3_client]):
        manager = SignedURLManager()
        client = manager._create_boto3_client()
        assert client == mock_s3_client
        assert manager._client_expiration_time is not None
        mock_sts_client.assume_role.assert_called_once_with(
            RoleArn=settings.AWS_SIGNING_ROLE_ARN,
            RoleSessionName="AssumeRoleSession",
        )


def test_refresh_client_if_needed_no_refresh() -> None:
    """Test for refreshing client when not needed."""
    mock_sts_client = create_mock_sts_client()

    manager = SignedURLManager()
    manager._client = MagicMock()
    manager._client_expiration_time = datetime.datetime.now(
        datetime.timezone.utc
    ) + datetime.timedelta(seconds=_PADDING_TIME_SECONDS + 100)
    manager.refresh_client_if_needed()
    mock_sts_client.assert_not_called()


def test_refresh_client_if_needed_refresh() -> None:
    """Test for refreshing client when needed."""
    mock_sts_client = create_mock_sts_client()
    mock_s3_client = MagicMock()

    with patch("boto3.client", side_effect=[mock_sts_client, mock_s3_client]):
        manager = SignedURLManager()
        manager._client = MagicMock()
        manager._client_expiration_time = datetime.datetime.now(
            datetime.timezone.utc
        ) + datetime.timedelta(seconds=_PADDING_TIME_SECONDS - 100)
        manager.refresh_client_if_needed()
        mock_sts_client.assume_role.assert_called_once_with(
            RoleArn="arn:aws:iam::123456789012:role/MyRole",
            RoleSessionName="AssumeRoleSession",
        )


def test_get_signed_url() -> None:
    """Test for getting signed URLs."""
    mock_sts_client = create_mock_sts_client()
    mock_s3_client = MagicMock()
    mock_generate_presigned_url = create_mock_generate_presigned_url()
    mock_s3_client.generate_presigned_url = mock_generate_presigned_url

    with patch("boto3.client", side_effect=[mock_sts_client, mock_s3_client]):
        manager = SignedURLManager()
        signed_url = manager.get_signed_url(
            "s3://my-bucket/my-file", enable_multi_region_access=True
        )
        assert signed_url == "https://example.com/presigned-url"
        mock_sts_client.assume_role.assert_called_once_with(
            RoleArn="arn:aws:iam::123456789012:role/MyRole",
            RoleSessionName="AssumeRoleSession",
        )
        mock_generate_presigned_url.assert_called_once_with(
            "get_object",
            Params={
                "Bucket": "arn:aws:s3::123456789012:accesspoint/testing.mrap",
                "Key": "my-file",
            },
            ExpiresIn=1800,
        )
