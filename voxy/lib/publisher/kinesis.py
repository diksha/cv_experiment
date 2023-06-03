from typing import Any, Callable, Dict, Optional, Union

import boto3
import botocore.exceptions as botoexcept

from lib.publisher.abstract import AbstractPublisher, PublishingError

DEFAULT_MAX_BUFFER_SIZE = 1000


class KinesisPartitionPublisher(AbstractPublisher):
    """Interface for publishing data to a specific stream and partition in Kinesis Data Streams"""

    def __init__(
        self,
        stream_name: str,
        partition_key: str,
        error_callback: Optional[
            Callable[
                [PublishingError],
                Any,
            ]
        ] = None,
        success_callback: Optional[Callable[[Dict[str, str]], Any]] = None,
        kinesis_client: Optional[boto3.client] = None,
        max_buffer_size: int = DEFAULT_MAX_BUFFER_SIZE,
    ):
        """
        Args:
            stream_name (str): name of the stream in Kinesis
            partition_key (str): key for data paritioning/sharding
            error_callback (Optional[Callable[[PublishingError], Any]]):
                Callback function to handle exceptions that cause dropped records
                Defaults to throwing error
            success_callback (Optional[Callable[[Dict[str, str]], Any]], optional):
                Callback function that takes the response of a successful PutRecord call
                Defaults to None.
            kinesis_client (boto3.client, optional):
                Override the default boto3 kinesis client to use for API calls
                Defaults to creating one with standard configuration.
            max_buffer_size (int, optional):
                The maximum number of records to store in memory before dropping additional records
                Defaults to 1000
        """
        super().__init__(max_buffer_size, error_callback, success_callback)

        self._stream_name = stream_name
        self._partition_key = partition_key

        self._kinesis_client = kinesis_client
        if self._kinesis_client is None:
            self._kinesis_client = boto3.client("kinesis")

        self._sequence_number: Optional[str] = None

    def put_datum(self, datum: Union[str, bytes]):
        """Queues a datum to be published asynchronously

        Raises:
            ValueError: if datum is None

        Args:
            datum (bytes | str): data to publish
        """
        super()._put_record(datum)

    def _handle_response(self, res: Dict[str, str]):
        self._sequence_number = res["SequenceNumber"]

    def _publish_record(self, data: bytes, _metadata) -> Dict[str, str]:
        """Sends a single datum to kinesis

        Args:
            data (bytes): data to put

        Returns:
            Dict[str, str]: response from Kinesis

        Raises:
            PublishingError: on failure to publish record
        """
        kwargs = {
            "StreamName": self._stream_name,
            "PartitionKey": self._partition_key,
            "Data": data,
        }

        if self._sequence_number is not None:
            kwargs["SequenceNumberForOrdering"] = self._sequence_number

        try:
            return self._kinesis_client.put_record(**kwargs)
        except (botoexcept.ClientError, botoexcept.BotoCoreError) as err:
            raise PublishingError(err) from err
