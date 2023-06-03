import unittest
from unittest.mock import Mock

from lib.publisher.sns import SNSPublisher


def get_sns_client() -> Mock:
    """
    Returns:
        Mock: mock SNS client
    """
    sns_client = Mock()
    sns_client.publish.return_value = {
        "ResponseMetadata": {
            "RequestId": "3d1e8d0c-1111-2222-3333-44b8e48f6d13"
        },
        "MessageId": "9048eb1d-1111-2222-3333-4b9f99b6184a",
    }
    return sns_client


class SNSPublisherTest(unittest.TestCase):
    def test_basic_publish(self):
        sns_client = get_sns_client()

        topic_arn = "arn:aws:sns:us-west-2:123456789012:my-topic"
        input_data = "message"

        publisher = SNSPublisher(
            topic_arn=topic_arn,
            sns_client=sns_client,
        )

        publisher.put_message(input_data)
        publisher.close()
        sns_client.publish.assert_called_once_with(
            TopicArn=topic_arn,
            MessageStructure="string",
            Message=input_data,
        )

    def test_message_group_id(self):
        sns_client = get_sns_client()

        topic_arn = "arn:aws:sns:us-west-2:123456789012:my-topic"
        message_group_id = "msg group"
        input_data = "message"

        publisher = SNSPublisher(
            topic_arn=topic_arn,
            sns_client=sns_client,
            message_group_id=message_group_id,
        )

        publisher.put_message(input_data)
        publisher.close()
        sns_client.publish.assert_called_once_with(
            TopicArn=topic_arn,
            MessageStructure="string",
            MessageGroupId=message_group_id,
            Message=input_data,
        )

    def test_deduplication_id(self):
        sns_client = get_sns_client()

        topic_arn = "arn:aws:sns:us-west-2:123456789012:my-topic"
        input_data = "message"
        deduplication_id = "deduplication_id"

        publisher = SNSPublisher(
            topic_arn=topic_arn,
            sns_client=sns_client,
        )

        publisher.put_message(input_data, deduplication_id)
        publisher.close()
        sns_client.publish.assert_called_once_with(
            TopicArn=topic_arn,
            MessageStructure="string",
            MessageDeduplicationId=deduplication_id,
            Message=input_data,
        )
