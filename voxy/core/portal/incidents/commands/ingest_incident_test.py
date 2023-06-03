from unittest.mock import MagicMock, patch

import pytest

from core.portal.incidents.commands import CreateIncident, IngestIncident
from core.structs.incident import Incident as IncidentStruct


@patch.object(CreateIncident, "execute")
@patch.object(CreateIncident, "__init__")
@pytest.mark.django_db
def test_new_incidents_passed_to_create_incident(
    create_incident_init_mock: MagicMock,
    create_incident_execute_mock: MagicMock,
) -> None:
    # Arrange
    data = IncidentStruct.from_dict({"uuid": "1234"})
    create_incident_init_mock.return_value = None

    # Act
    IngestIncident(data).execute()

    # Assert
    create_incident_init_mock.assert_called_once_with(data)
    create_incident_execute_mock.assert_called_once()
