from unittest.mock import MagicMock, patch

import pytest

from core.portal.lib.utils.signed_url_manager import signed_url_manager
from core.portal.testing.factories import CameraFactory, IncidentFactory


@patch.object(signed_url_manager, "get_signed_url")
@pytest.mark.django_db
def test_thumbnail_url_uses_saved_s3_path_when_available(
    get_signed_url_mock: MagicMock,
) -> None:
    """Test that s3 path is used for thumbnail when available.

    Args:
        get_signed_url_mock (MagicMock): get_signed_url mock
    """
    fake_signed_url = "https://example.com/image.jpg"
    get_signed_url_mock.return_value = fake_signed_url
    camera = CameraFactory(thumbnail_s3_path="foo")

    assert camera.thumbnail_url == fake_signed_url
    get_signed_url_mock.assert_called_with(s3_path=camera.thumbnail_s3_path)


@patch.object(signed_url_manager, "get_signed_url")
@pytest.mark.django_db
def test_thumbnail_url_saves_new_s3_path_from_latest_incident(
    get_signed_url_mock: MagicMock,
) -> None:
    """Test that incident thumbnail is used/saved if camera doesn't have thumbnail.

    Args:
        get_signed_url_mock (MagicMock): get_signed_url mock
    """
    get_signed_url_mock.return_value = "https://example.com/image.jpg"
    camera = CameraFactory(thumbnail_s3_path=None)
    incident = IncidentFactory(
        camera=camera,
        data={"video_thumbnail_s3_path": "s3://example-bucket/image.jpg"},
    )

    # accessing .thumbnail_url executes the logic under test
    camera_thumbnail_url = camera.thumbnail_url

    assert "https" in camera_thumbnail_url
    assert incident.video_thumbnail_s3_path is not None
    assert camera.thumbnail_s3_path == incident.video_thumbnail_s3_path
    get_signed_url_mock.assert_called_with(
        s3_path=incident.video_thumbnail_s3_path,
    )


@pytest.mark.django_db
def test_thumbnail_url_returns_none_when_thumbnail_not_available() -> None:
    """Test that thumbnail_url is None if no thumbnail is available."""
    camera = CameraFactory(thumbnail_s3_path=None)
    assert camera.thumbnail_url is None
