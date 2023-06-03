import unittest

from core.execution.nodes.publisher import PublisherNode


class PublisherNodeTest(unittest.TestCase):
    def test_import(self) -> None:
        """
        Tests to see if the import failed
        """

        self.assertTrue(PublisherNode is not None)
