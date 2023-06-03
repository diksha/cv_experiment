import unittest

# trunk-ignore-all(pylint/E0611): ignore pb import errors
from protos.platform.bowser.v1.bowser_config_keys_pb2 import Unit
from services.platform.bowser.engine.utils.key_utils import KeyUtils


class KeyUtilsTest(unittest.TestCase):
    def test_1_modulo_second(self):
        self.assertEqual(
            ["2023-04-06", "23:36:12"],
            KeyUtils.get_timestamp_granularity(1680824172000, 1),
        )

    def test_2_modulo_minute(self):
        self.assertEqual(
            ["2023-04-06", "23:36:00"],
            KeyUtils.get_timestamp_granularity(1680824172000, 60),
        )
        self.assertEqual(
            ["2023-04-06", "23:36:00"],
            KeyUtils.get_timestamp_granularity_by_proto_unit(
                1680824172000, Unit.UNIT_MINUTE_UNSPECIFIED
            ),
        )

    def test_3_modulo_hours(self):
        self.assertEqual(
            ["2023-04-06", "23:00:00"],
            KeyUtils.get_timestamp_granularity(1680824172000, 3600),
        )
        self.assertEqual(
            ["2023-04-06", "23:00:00"],
            KeyUtils.get_timestamp_granularity_by_proto_unit(
                1680824172000, Unit.UNIT_HOURS
            ),
        )

    def test_4_modulo_day(self):
        self.assertEqual(
            "2023-04-06",
            KeyUtils.get_timestamp_granularity(1680824172000, 86400)[0],
        )
        self.assertEqual(
            "2023-04-06",
            KeyUtils.get_timestamp_granularity_by_proto_unit(
                1680824172000, Unit.UNIT_DAY
            )[0],
        )

    def test_5_modulo_week(self):
        self.assertEqual(
            ["2023-04-03"],
            KeyUtils.get_timestamp_granularity_by_proto_unit(
                1680824172000, Unit.UNIT_WEEK
            ),
        )

    def test_6_modulo_30_day(self):
        self.assertEqual(
            ["2023-04-01"],
            KeyUtils.get_timestamp_granularity_by_proto_unit(
                1680824172000, Unit.UNIT_MONTH
            ),
        )
