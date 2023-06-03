from typing import Any, Callable, Dict, Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from lib.publisher.abstract import AbstractPublisher, PublishingError

DEFAULT_MAX_BUFFER_SIZE = 1000
METADATA_KEY_DEDUPLICATION = "dedupe_key"


class SNSPublisher(AbstractPublisher):
    """Interface for publishing data to a specific stream and partition in Kinesis Data Streams"""

    def __init__(
        self,
        topic_arn: str,
        message_group_id: str = None,
        error_callback: Callable[
            [PublishingError],
            Any,
        ] = None,
        success_callback: Optional[Callable[[Dict[str, str]], Any]] = None,
        sns_client: Optional[boto3.client] = None,
        max_buffer_size: int = DEFAULT_MAX_BUFFER_SIZE,
    ):
        """
        Args:
            topic_arn (str): Amazon Resource Name for the SNS topic
            message_group_id (str, optional):
                Key for group in which ordering is guaranteed.
                Required for FIFO topics, disallowed for standard topics
                Defaults to None
            error_callback (Callable[[PublishingError], Any], optional):
                Callback function to handle exceptions that cause dropped records
                Defaults to throwing errors
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
        self._topic_arn = topic_arn
        self._message_group_id = message_group_id

        self._sns_client = sns_client
        if self._sns_client is None:
            self._sns_client = boto3.client("sns")

    def put_message(
        self, message: str, deduplication_key: Optional[str] = None
    ) -> None:
        """Put a single datum to SNS. It will be added to queue and published asynchronously

        Args:
            message (str): a string message to put to SNS
            deduplication_key (str, optional):
                Messages with an identical key to one stored will be dropped
                Defaults to None

        Raises:
            ValueError: if message is None
        """

        metadata = None
        if deduplication_key is not None:
            metadata = {METADATA_KEY_DEDUPLICATION: deduplication_key}

        super()._put_record(record=message, metadata=metadata)

    def _handle_response(self, res) -> None:
        pass

    def _publish_record(self, data: Any, metadata: Optional[Dict]):
        try:
            args = {
                "TopicArn": self._topic_arn,
                "MessageStructure": "string",
                "Message": data,
            }

            if self._message_group_id is not None:
                args["MessageGroupId"] = self._message_group_id

            if metadata is not None:
                args["MessageDeduplicationId"] = metadata[
                    METADATA_KEY_DEDUPLICATION
                ]

            return self._sns_client.publish(**args)
        except (ClientError, BotoCoreError) as err:
            raise PublishingError(err) from err
