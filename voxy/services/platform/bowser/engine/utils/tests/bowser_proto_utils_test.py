import unittest

from protos.platform.bowser.v1.bowser_config_keys_pb2 import Unit
from services.platform.bowser.engine.utils.bowser_proto_utils import (
    BowserProtoUtils,
)

# trunk-ignore-all(pylint/E0611): ignore pb import errors


class BowserProtoUtilsTest(unittest.TestCase):
    def test_8_create_key_proto(self):
        tuple_key = ("key1", "key2", "key3")
        timestamp = "timestamp_key"
        unit = Unit.UNIT_DAY
        proto = BowserProtoUtils.create_key_proto(tuple_key, timestamp, unit)
        self.assertEqual(len(proto.fields), len(tuple_key))
        self.assertEqual(proto.timestamp.field, timestamp)
        self.assertEqual(proto.timestamp.by, unit)

    def test_9_create_s3_proto(self):
        proto = BowserProtoUtils.create_s3_consumer_proto(
            "bucket_1", ("uri_1", "uri_2")
        )
        self.assertEqual(proto.name, "S3 Consumer")
        self.assertEqual(len(proto.aws.s3.buckets), 1)
        self.assertEqual(len(proto.aws.s3.buckets[0].uris), 2)

    def test_10_create_windo_proto(self):
        proto = BowserProtoUtils.create_window_proto(1)
        self.assertEqual(proto.time_second, 1)
