import unittest
from unittest.mock import Mock, call

from lib.publisher.kinesis import KinesisPartitionPublisher


class KinesisPartitionPublisherTest(unittest.TestCase):
    def test_publish_params(self):
        kinesis_client = Mock()
        kinesis_client.put_record.return_value = {
            "ShardId": "shardId-000000000000",
            "SequenceNumber": "49545115243490985018280067714973144582180062593244200961",
            "EncryptionType": "NONE",
        }

        stream_name = "test_stream_name"
        partition_key = "test_partition_key"
        input_data = "test_input"

        expected_params = {
            "StreamName": stream_name,
            "PartitionKey": partition_key,
            "Data": input_data,
        }

        publisher = KinesisPartitionPublisher(
            stream_name=stream_name,
            partition_key=partition_key,
            kinesis_client=kinesis_client,
        )

        publisher.put_datum(input_data)
        publisher.close()
        kinesis_client.put_record.assert_called_once_with(**expected_params)

    def test_update_sequence_id(self):
        kinesis_client = Mock()
        kinesis_client.put_record.return_value = {
            "ShardId": "shardId-000000000000",
            "SequenceNumber": "12345",
            "EncryptionType": "NONE",
        }

        stream_name = "test_stream_name"
        partition_key = "test_partition_key"
        input_data = ["test_input_1", "test_input_2"]

        publisher = KinesisPartitionPublisher(
            stream_name=stream_name,
            partition_key=partition_key,
            kinesis_client=kinesis_client,
        )

        publisher.put_datum(input_data[0])
        publisher.put_datum(input_data[1])
        publisher.close()

        assert kinesis_client.put_record.call_args_list == [
            call(
                StreamName=stream_name,
                PartitionKey=partition_key,
                Data=input_data[0],
            ),
            call(
                StreamName=stream_name,
                PartitionKey=partition_key,
                Data=input_data[1],
                SequenceNumberForOrdering="12345",
            ),
        ]
