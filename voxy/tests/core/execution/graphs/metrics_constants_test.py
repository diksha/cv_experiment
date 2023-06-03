import unittest

from core.execution.graphs.metrics_constants import (
    AttributeNames,
    MetricNames,
    SpanNames,
)


class MetricsConstantsTest(unittest.TestCase):
    """
    Tests the metrics constants file.
    """

    def _get_all_variables_in_the_class(self, class_obj):
        return [
            value
            for name, value in vars(class_obj).items()
            if not callable(value)
            and not name.startswith("__")
            and isinstance(value, str)
        ]

    def test_constants_exists(self) -> None:
        self.assertTrue(
            len(self._get_all_variables_in_the_class(MetricNames)) != 0
        )

        self.assertTrue(
            len(self._get_all_variables_in_the_class(AttributeNames)) != 0
        )

        self.assertTrue(
            len(self._get_all_variables_in_the_class(SpanNames)) != 0
        )
