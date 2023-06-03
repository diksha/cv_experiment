from google.protobuf.internal.enum_type_wrapper import EnumTypeWrapper
from google.protobuf.pyext._message import FieldDescriptor

# trunk-ignore-all(pylint/C0301): ignore pb import errors


class ProtoException(Exception):
    def __init__(self, *args: object) -> None:
        """Exception to raise a protobuf util error

        Args:
            *args: Exception args
        :raises ProtoException: when a proto config error happens


        """
        super().__init__(*args)


class ProtoUtils:
    @staticmethod
    def get_extension_by_enum(
        proto_enum_entity: EnumTypeWrapper,
        proto_extension: FieldDescriptor,
        enum,
    ):
        """Take a Proto Enum and check if it is owning and extension

        :param EnumTypeWrapper proto_enum_entity: Any proto enum
        :param FieldDescriptor proto_extension: Field of an EnumValueOptions
        :param Integer enum: The actual Enum Integer that you want to get the extension for
        :raises ProtoException : When a config error happens
        :returns: the extension value type
        :rtype: Generic

        """

        if (
            proto_enum_entity is None
            or proto_extension is None
            or enum is None
        ):
            raise ProtoException(
                "You can not have None value for any of the parameters"
            )

        if proto_enum_entity.DESCRIPTOR is None:
            raise ProtoException(
                f"Your proto_enum_entity {str(proto_enum_entity)} is not a proper Proto Enum"
            )

        if proto_enum_entity.DESCRIPTOR.values_by_number[enum] is None:
            raise ProtoException(
                f"The Proto enum {enum} value does not exist inside the Enum {str(proto_enum_entity)}"
            )

        if (
            proto_enum_entity.DESCRIPTOR.values_by_number[enum].GetOptions()
            is None
        ):
            raise ProtoException(
                f"The Proto enum value {str(proto_enum_entity)} does not embed an Extensions"
            )

        if (
            proto_enum_entity.DESCRIPTOR.values_by_number[enum]
            .GetOptions()
            .Extensions[proto_extension]
            is None
        ):
            raise ProtoException(
                f"The Proto enum value {str(proto_enum_entity)} does not embed an Extensions named {proto_extension}"
            )

        value_to_return = (
            proto_enum_entity.DESCRIPTOR.values[enum]
            .GetOptions()
            .Extensions[proto_extension]
        )

        if value_to_return is None:
            raise ProtoException("The extension value is empty or None")

        return value_to_return
