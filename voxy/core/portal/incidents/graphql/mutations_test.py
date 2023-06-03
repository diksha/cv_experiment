import mock
import pytest

from core.portal.api.models.incident import Incident
from core.portal.incidents.graphql.mutations import (
    IncidentReopen,
    IncidentResolve,
)
from core.portal.testing.factories import IncidentFactory


@pytest.mark.django_db
@mock.patch.object(IncidentResolve, "send_email")
@mock.patch("core.portal.incidents.graphql.mutations.log_action")
@mock.patch("core.portal.incidents.graphql.mutations.pk_from_global_id")
@mock.patch("core.portal.incidents.graphql.mutations.has_zone_permission")
def test_incident_resolve_mutation(
    mock_has_zone_permission,
    mock_pk_from_global_id,
    mock_log_action,
    mock_send_email,
) -> None:
    """Test that incident resolve mutation returns expected values and has expected side effects.

    Args:
        mock_has_zone_permission (MagicMock): has_zone_permission mock
        mock_pk_from_global_id (MagicMock): pk_from_global_id mock
        mock_log_action (MagicMock): log_action mock
        mock_send_email (MagicMock): send_email mock
    """
    IncidentFactory()
    inc = Incident.objects_raw.first()

    mock_pk_from_global_id.return_value = (None, inc.pk)
    mock_has_zone_permission.return_value = True

    res = IncidentResolve.mutate(
        root=None, info=mock.Mock(), incident_id=inc.pk
    )

    incident = Incident.objects.get(pk=inc.pk)

    assert isinstance(res, IncidentResolve)
    assert mock_has_zone_permission.call_count == 1
    assert mock_send_email.call_count == 1
    assert mock_log_action.call_count == 1
    assert incident.status == Incident.Status.RESOLVED


@pytest.mark.django_db
@mock.patch("core.portal.incidents.graphql.mutations.log_action")
@mock.patch("core.portal.incidents.graphql.mutations.pk_from_global_id")
def test_incident_reopen_mutation(
    mock_pk_from_global_id, mock_log_action
) -> None:
    """Test that incident reopen mutation returns expected values and has expected side effects.

    Args:
        mock_pk_from_global_id (MagicMock): pk_from_global_id mock
        mock_log_action (MagicMock): log_action mock
    """
    IncidentFactory(
        status=Incident.Status.RESOLVED,
    )
    inc = Incident.objects_raw.first()

    mock_pk_from_global_id.return_value = (None, inc.pk)

    res = IncidentReopen.mutate(
        root=None, info=mock.Mock(), incident_id=inc.pk
    )

    incident = Incident.objects.get(pk=inc.pk)

    assert isinstance(res, IncidentReopen)
    assert mock_log_action.call_count == 1
    assert incident.status == Incident.Status.OPEN
