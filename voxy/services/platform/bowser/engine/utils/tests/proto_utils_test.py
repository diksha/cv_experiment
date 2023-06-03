import unittest

from protos.platform.bowser.v1.bowser_config_keys_pb2 import Unit, seconds
from services.platform.bowser.engine.utils.proto_utils import (
    ProtoException,
    ProtoUtils,
)

# trunk-ignore-all(pylint/E0611): ignore pb import errors


class ProtoUtilsTest(unittest.TestCase):
    def test_1_all_arg_none(self):
        with self.assertRaises(ProtoException):
            raise ProtoUtils.get_extension_by_enum(None, None, None)

    def test_2_only_proto_entity_is_not_null(self):
        with self.assertRaises(ProtoException):
            raise ProtoUtils.get_extension_by_enum(Unit, None, None)

    def test_3_only_proto_extension_is_not_null(self):
        with self.assertRaises(ProtoException):
            raise ProtoUtils.get_extension_by_enum(None, seconds, None)

    def test_3_only_enum_value_is_not_null(self):
        with self.assertRaises(ProtoException):
            raise ProtoUtils.get_extension_by_enum(None, None, "UNIT_HOURS")

    def test_4_proto_entity_is_not_a_proto_enum(self):
        with self.assertRaises(AttributeError):
            raise ProtoUtils.get_extension_by_enum(
                ProtoUtilsTest, seconds, "UNIT_HOURS"
            )

    def test_5_enum_value_does_not_exist_self(self):
        with self.assertRaises(KeyError):
            raise ProtoUtils.get_extension_by_enum(
                Unit, "NOT_A_PROTO_FIELD", "NO_A_ENUM_STRING"
            )

    def test_6_enum_extension_is_not_a_proto_field(self):
        with self.assertRaises(KeyError):
            raise ProtoUtils.get_extension_by_enum(
                Unit, "NOT_A_PROTO_FIELD", "UNIT_HOURS"
            )

    def test_7_all_good(self):
        self.assertEqual(
            3600,
            ProtoUtils.get_extension_by_enum(Unit, seconds, Unit.UNIT_HOURS),
        )
