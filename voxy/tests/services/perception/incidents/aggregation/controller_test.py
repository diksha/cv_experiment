import unittest

from core.structs.incident import Incident
from services.perception.incidents.aggregation.controller import (
    IncidentAggregationController,
)


class TestIncidentAggregationController(unittest.TestCase):
    def setUp(self):
        self._controller = IncidentAggregationController()

    def get_base_incident(self, uuid, incident_type_id) -> Incident:
        """Generate a base incident.

        Args:
            uuid (_type_): incident uuid
            incident_type_id (_type_): incident type id

        Returns:
            Incident: incident
        """
        incident = Incident()
        incident.incident_type_id = "PRODUCTION_LINE_DOWN"
        incident.run_uuid = "run_uuid"
        incident.camera_uuid = "camera_uuid"
        incident.track_uuid = "track_uuid"
        incident.uuid = uuid
        return incident

    def test_aggregation(self):
        incident_1 = self.get_base_incident("uuid1", "PRODUCTION_LINE_DOWN")
        incident_1.start_frame_relative_ms = 0
        incident_1.end_frame_relative_ms = 1000
        incident_1.sequence_id = 0

        output = self._controller.process(incident_1)
        self.assertEqual(output.uuid, "uuid1")

        incident_2 = self.get_base_incident("uuid2", "PRODUCTION_LINE_DOWN")
        incident_2.start_frame_relative_ms = 1000
        incident_2.end_frame_relative_ms = 2000
        incident_2.sequence_id = 0
        incident_2.cooldown_tag = True

        output = self._controller.process(incident_2)
        self.assertEqual(output.uuid, "uuid1")
        self.assertEqual(output.tail_incident_uuids, ["uuid2"])
        self.assertEqual(output.end_frame_relative_ms, 2000)

        incident_3 = self.get_base_incident("uuid3", "PRODUCTION_LINE_DOWN")
        incident_3.start_frame_relative_ms = 2000
        incident_3.end_frame_relative_ms = 3000
        incident_3.sequence_id = 0
        incident_3.cooldown_tag = True

        output = self._controller.process(incident_3)

        self.assertEqual(output.uuid, "uuid1")
        self.assertEqual(output.tail_incident_uuids, ["uuid2", "uuid3"])
        self.assertEqual(output.end_frame_relative_ms, 3000)

    def test_non_cooldown(self):
        incident_1 = self.get_base_incident("uuid1", "PRODUCTION_LINE_DOWN")
        incident_1.start_frame_relative_ms = 0
        incident_1.end_frame_relative_ms = 1000
        incident_1.sequence_id = 0

        output = self._controller.process(incident_1)
        self.assertEqual(output.uuid, "uuid1")

        incident_2 = self.get_base_incident("uuid2", "PRODUCTION_LINE_DOWN")
        incident_2.start_frame_relative_ms = 1000
        incident_2.end_frame_relative_ms = 2000
        incident_2.sequence_id = 0

        output = self._controller.process(incident_2)
        self.assertEqual(output.uuid, "uuid2")

    def test_non_aggregation_incident_type(self):
        incident_1 = self.get_base_incident("uuid1", "BAD_POSTURE")
        incident_1.sequence_id = 0
        output = self._controller.process(incident_1)
        self.assertEqual(output.uuid, "uuid1")
