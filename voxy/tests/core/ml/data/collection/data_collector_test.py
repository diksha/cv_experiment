import unittest

from core.ml.data.collection.data_collector import DataFlywheelCollector


class TestDataCollector(unittest.TestCase):
    """Tests collecting incident from portal

    Args:
        unittest (TestCase): Unit Teset
    """

    def setUp(self) -> None:
        self.data_collector = DataFlywheelCollector()

    def test_execute_query(self) -> None:
        """Tests executing a query"""
        # Check collecting invalid
        uuids = self.data_collector.execute_query(
            "open_door_duration", "invalid", max_num_incidents=1
        )
        self.assertTrue(uuids)

        # Check collecting valid
        uuids = self.data_collector.execute_query(
            "safety_vest", "valid", max_num_incidents=1
        )
        self.assertTrue(uuids)

        # Check collecting all
        uuids = self.data_collector.execute_query(
            "piggyback", "all", max_num_incidents=1
        )
        self.assertTrue(uuids)
