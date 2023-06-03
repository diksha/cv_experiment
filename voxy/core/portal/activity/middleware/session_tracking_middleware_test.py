import pytest
from django.contrib.auth.models import AnonymousUser
from django.http import HttpRequest
from mock import MagicMock, call, patch

from core.portal.activity.middleware import SessionTrackingMiddleware
from core.portal.testing.factories import (
    OrganizationFactory,
    UserFactory,
    ZoneFactory,
)


@patch.object(SessionTrackingMiddleware, "track_session")
def test_call_does_not_track_if_user_is_none(mock_track_session: MagicMock):
    """Test that middleware skips missing users."""
    # Arrange
    request = HttpRequest()
    request.user = None
    middleware = SessionTrackingMiddleware(get_response=lambda _: None)

    # Act
    middleware(request)

    # Assert
    assert mock_track_session.call_count == 0


@patch.object(SessionTrackingMiddleware, "track_session")
def test_call_does_not_track_if_anonymous_user(mock_track_session: MagicMock):
    """Test that middleware skips anonymous users."""
    # Arrange
    request = HttpRequest()
    request.user = AnonymousUser()
    middleware = SessionTrackingMiddleware(get_response=lambda _: None)

    # Act
    middleware(request)

    # Assert
    assert mock_track_session.call_count == 0


@pytest.mark.django_db
@patch("django.core.cache.cache.set_many")
@patch("time.time")
def test_track_session_tracks_available_scopes_all(
    mock_time: MagicMock,
    mock_cache_set_many: MagicMock,
):
    """Test that middleware tracks all scopes when available."""
    # Arrange
    mock_time.return_value = 123
    org = OrganizationFactory()
    site = ZoneFactory(organization=org)
    user = UserFactory()
    user.profile.organization = org
    user.profile.site = site
    middleware = SessionTrackingMiddleware(get_response=lambda _: None)

    # Act
    middleware.track_session(user)

    # Assert
    expected_active_entries = {
        "user_session:active:1:1:GLOBAL": "123:123",
        f"user_session:active:1:2:{org.id}": "123:123",
        f"user_session:active:1:3:{site.id}": "123:123",
    }
    expected_etl_entries = {
        "user_session:etl:1:1:GLOBAL:123": 123,
        f"user_session:etl:1:2:{org.id}:123": 123,
        f"user_session:etl:1:3:{site.id}:123": 123,
    }
    mock_cache_set_many.assert_has_calls(
        [
            call(expected_active_entries, timeout=900),
            call(expected_etl_entries, timeout=None),
        ]
    )


@pytest.mark.django_db
@patch("django.core.cache.cache.set_many")
@patch("time.time")
def test_track_session_tracks_available_scopes_global(
    mock_time: MagicMock,
    mock_cache_set_many: MagicMock,
):
    """Test that middleware tracks global scope when org/site not available."""
    # Arrange
    mock_time.return_value = 123
    user = UserFactory()
    user.profile.organization = None
    user.profile.site = None
    middleware = SessionTrackingMiddleware(get_response=lambda _: None)

    # Act
    middleware.track_session(user)

    # Assert
    expected_active_entries = {
        "user_session:active:1:1:GLOBAL": "123:123",
    }
    expected_etl_entries = {
        "user_session:etl:1:1:GLOBAL:123": 123,
    }
    mock_cache_set_many.assert_has_calls(
        [
            call(expected_active_entries, timeout=900),
            call(expected_etl_entries, timeout=None),
        ]
    )


@pytest.mark.django_db
@patch("django.core.cache.cache.set_many")
@patch("time.time")
def test_track_session_tracks_available_scopes_global_org(
    mock_time: MagicMock,
    mock_cache_set_many: MagicMock,
):
    """Test that middleware tracks global/org scope when site not available."""
    # Arrange
    mock_time.return_value = 123
    org = OrganizationFactory()
    user = UserFactory()
    user.profile.organization = org
    user.profile.site = None
    middleware = SessionTrackingMiddleware(get_response=lambda _: None)

    # Act
    middleware.track_session(user)

    # Assert
    expected_active_entries = {
        "user_session:active:1:1:GLOBAL": "123:123",
        f"user_session:active:1:2:{org.id}": "123:123",
    }
    expected_etl_entries = {
        "user_session:etl:1:1:GLOBAL:123": 123,
        f"user_session:etl:1:2:{org.id}:123": 123,
    }
    mock_cache_set_many.assert_has_calls(
        [
            call(expected_active_entries, timeout=900),
            call(expected_etl_entries, timeout=None),
        ]
    )
