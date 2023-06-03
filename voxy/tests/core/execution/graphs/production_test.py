import unittest

from core.execution.graphs.production import ProductionGraph


class ProductionGraphTest(unittest.TestCase):
    """
    Tests production graph
    """

    def test_import(self) -> None:
        self.assertTrue(ProductionGraph is not None)
