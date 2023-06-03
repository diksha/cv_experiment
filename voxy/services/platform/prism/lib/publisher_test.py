import unittest
from unittest.mock import Mock, call, patch

import botocore

from services.platform.prism.lib.publisher import PrismPublisher

_SNS_MODEL = botocore.session.get_session().get_service_model("sns")
_SNS_FACTORY = botocore.errorfactory.ClientExceptionsFactory()
_SNS_EXCEPTIONS = _SNS_FACTORY.create_client_exceptions(_SNS_MODEL)

# trunk-ignore-all(pylint/C0301): don't care about line too long


class PrismPublisherTest(unittest.TestCase):
    def test_publish_incidents(self):
        camera_uuid = "mock_camera_uuid"
        kvs_stream_arn = "mock_stream_arn"
        sns_topic_arn = "mock_topic_arn"
        metrics_client = Mock()

        mock_sns_client = Mock()
        mock_boto3 = Mock()
        mock_boto3.client.return_value = mock_sns_client

        mock_sns_client.publish.side_effect = [
            "success_response",
            _SNS_EXCEPTIONS.AuthorizationErrorException({}, ""),
            "success_response",
        ]

        incidents_data = [
            {
                "uuid": "success1_incident_uuid",
                "key1": "value1",
            },
            {
                "uuid": "failure1_incident_uuid",
                "key2": "value2",
            },
            {
                "uuid": "success2_incident_uuid",
                "key3": "value3",
            },
        ]

        incidents = []
        for incident_data in incidents_data:
            incident = Mock()
            incident.to_dict.return_value = incident_data
            incident.uuid = incident_data["uuid"]
            incidents.append(incident)

        with patch("lib.publisher.sns.boto3", mock_boto3):
            publisher = PrismPublisher.from_config_values(
                camera_uuid=camera_uuid,
                kvs_stream_arn=kvs_stream_arn,
                sns_topic_arn=sns_topic_arn,
                metrics_client=metrics_client,
            )

        for incident in incidents:
            publisher.publish_incident(incident)
        publisher.close()

        assert mock_sns_client.publish.call_count == 3
        mock_sns_client.publish.assert_has_calls(
            [
                call(
                    TopicArn=sns_topic_arn,
                    MessageStructure="string",
                    Message='{"uuid": "success1_incident_uuid", "key1": "value1", "kvs_stream_arn": "mock_stream_arn"}',
                    MessageGroupId="mock_camera_uuid",
                    MessageDeduplicationId="success1_incident_uuid",
                ),
                call(
                    TopicArn=sns_topic_arn,
                    MessageStructure="string",
                    Message='{"uuid": "failure1_incident_uuid", "key2": "value2", "kvs_stream_arn": "mock_stream_arn"}',
                    MessageGroupId="mock_camera_uuid",
                    MessageDeduplicationId="failure1_incident_uuid",
                ),
                call(
                    TopicArn=sns_topic_arn,
                    MessageStructure="string",
                    Message='{"uuid": "success2_incident_uuid", "key3": "value3", "kvs_stream_arn": "mock_stream_arn"}',
                    MessageGroupId="mock_camera_uuid",
                    MessageDeduplicationId="success2_incident_uuid",
                ),
            ]
        )

        metrics_client.increment_metric_counter.assert_has_calls(
            [
                call("incident_published", 1),
                call("incident_dropped", 1),
                call("incident_published", 1),
            ]
        )
