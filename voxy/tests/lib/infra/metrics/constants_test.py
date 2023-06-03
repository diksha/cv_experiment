import unittest

from lib.infra.metrics.constants import CommonConstants, Environment


class MetricsConstantsTest(unittest.TestCase):
    """
    Tests the metrics constants file.
    """

    def _get_all_variables_in_the_class(self, class_obj):
        return [
            value
            for name, value in vars(class_obj).items()
            if not callable(value) and not name.startswith("__")
        ]

    def test_common_constants_exists(self) -> None:
        self.assertTrue(
            len(self._get_all_variables_in_the_class(Environment)) != 0
        )

        self.assertTrue(
            len(self._get_all_variables_in_the_class(CommonConstants)) != 0
        )
