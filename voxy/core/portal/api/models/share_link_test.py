import pytest
from django.conf import settings
from django.utils import timezone

from core.portal.api.models.share_link import ShareLink
from core.portal.testing.factories import IncidentFactory, UserFactory


@pytest.mark.django_db
def test_generate_share_links_correctly() -> None:
    """Test that share links are generated correctly."""

    shared_by = UserFactory()
    incident = IncidentFactory()

    generated_share_link = ShareLink.generate(
        shared_by=shared_by,
        incident=incident,
    )

    token = generated_share_link.split(f"{settings.BASE_URL}/share/")[1]
    share_link = ShareLink.objects.get(token=token)

    assert share_link.expires_at > timezone.now()
