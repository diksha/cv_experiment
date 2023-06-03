from typing import Tuple

from protos.platform.bowser.v1.bowser_config_consumer_pb2 import (
    ProcessorConsumer,
    ProcessorConsumerAwsS3Bucket,
)
from protos.platform.bowser.v1.bowser_config_function_pb2 import (
    ProcessorFunction,
)
from protos.platform.bowser.v1.bowser_config_keys_pb2 import (
    ProcessorKeys,
    Unit,
)
from protos.platform.bowser.v1.bowser_config_processor_pb2 import Processor
from protos.platform.bowser.v1.bowser_config_sink_pb2 import ProcessorSink
from protos.platform.bowser.v1.bowser_config_window_pb2 import ProcessorWindow
from protos.platform.bowser.v1.bowser_env_config_pb2 import (
    BowserEnv,
    BowserEnvMode,
)

# trunk-ignore-all(pylint/E0611): ignore pb import errors


class BowserProtoUtils:
    @staticmethod
    def create_key_proto(
        tuple_key: Tuple,
        timestamp_mapping_key: str = None,
        timestamp_unit_group_by: Unit = None,
    ) -> ProcessorKeys:
        """Create a Keys Proto for a Bowser Processor


        :param Tuple tuple_key: Representing the key fields
        :param str timestamp_mapping_key: Str. Representing the timestamp key field
        :param Unit timestamp_unit_group_by: Representing the Unit of the timestamp to use
        :returns: Protobuf config
        :rtype: ProcessorKeys


        """
        proto = ProcessorKeys()
        proto.fields.extend(tuple_key)

        if (
            timestamp_mapping_key is not None
            and timestamp_unit_group_by is not None
        ):
            proto.timestamp.field = timestamp_mapping_key
            proto.timestamp.by = timestamp_unit_group_by
        return proto

    @staticmethod
    def create_s3_consumer_proto(
        s3_bucket: str,
        s3_bucket_uris: list,
        consumer_name: str = "S3 Consumer",
    ) -> ProcessorConsumer:
        """Create a S3 Consumer proto config

        :param str s3_bucket: Name of the bucket to consume the S3 file
        :param list s3_bucket_uris:  Uris folder containing the file to consume
        :param str consumer_name: Name of the Consumer

        :returns: Protobuf config
        :rtype: ProcessorConsumer

        """
        proto = ProcessorConsumer()
        proto.name = consumer_name
        proto_buckets = []
        proto_bucket = ProcessorConsumerAwsS3Bucket()
        proto_bucket.name = s3_bucket
        proto_bucket.uris.extend(s3_bucket_uris)
        proto_buckets.append(proto_bucket)
        proto.aws.s3.buckets.extend(proto_buckets)
        return proto

    @staticmethod
    def create_window_proto(time_second: int = 30) -> ProcessorWindow:
        """Create a Window Proto config

        :param int time_second: maximum time period in second of the window

        :returns: Protobuf config
        :rtype: ProcessorWindow

        """
        proto = ProcessorWindow()
        proto.time_second = time_second
        return proto

    @staticmethod
    def create_sink_local_proto(
        sink_name: str, out_location: str
    ) -> ProcessorSink:
        """Create a Processor Sink Proto Config

        :param str sink_name: the name of the config
        :param str out_location: the location to where generate the local file

        :returns: Protobuf config
        :rtype: ProcessorSink

        """
        proto = ProcessorSink()
        proto.name = sink_name
        proto.local.out_location = out_location
        return proto

    @staticmethod
    def create_bowser_env_proto(
        mode: BowserEnvMode, max_parrallelism: int
    ) -> BowserEnv:
        """

        :param BowserEnvMode mode: BATCH or STREAM
        :param int max_parrallelism: an integer that capp the maximum parallelism

        :returns: Protobuf config
        :rtype: BowserEnv

        """
        proto = BowserEnv()
        proto.mode = mode
        proto.max_parallelism = max_parrallelism
        return proto

    @staticmethod
    def create_processor_proto(
        processor_name: str,
        consumer: ProcessorConsumer,
        keys: ProcessorKeys,
        window: ProcessorWindow,
        function: ProcessorFunction,
        sink: ProcessorSink,
    ) -> Processor:
        """Create a Processor Proto Config


        :param str processor_name: processor's name
        :param ProcessorConsumer consumer: processor consumer config
        :param ProcessorKeys keys: key by processor proto config
        :param ProcessorWindow window: processor window proto config
        :param ProcessorFunction function: processor function proto config
        :param ProcessorSink sink: processor sink proto config

        :returns: Protobuf config
        :rtype: Processor

        """
        proto = Processor()
        proto.name = processor_name
        proto.consumer.CopyFrom(consumer)
        proto.key.CopyFrom(keys)
        proto.window.CopyFrom(window)
        proto.function.CopyFrom(function)
        proto.sink.CopyFrom(sink)
        return proto
