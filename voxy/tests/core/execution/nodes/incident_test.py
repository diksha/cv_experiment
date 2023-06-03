import unittest

from core.execution.nodes.incident_writer import IncidentWriterNode


class IncidentWriterNodeTest(unittest.TestCase):
    """
    Tests incident node
    """

    def test_import(self) -> None:
        """
        Tests incident node
        """

        self.assertTrue(IncidentWriterNode is not None)
