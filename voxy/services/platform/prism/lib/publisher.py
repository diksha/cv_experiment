# Copyright 2022 Voxel Labs, Inc.
# All rights reserved.

# This document may not be reproduced, republished, distributed, transmitted,
# displayed, broadcast or otherwise exploited in any manner without the express
# prior written permission of Voxel Labs, Inc. The receipt or possession of this
# document does not convey any rights to reproduce, disclose, or distribute its
# contents, or to manufacture, use, or sell anything that it may describe, in
# whole or in part.

import atexit
import json
from typing import Iterable, Type

from loguru import logger

from core.structs.incident import Incident
from lib.infra.metrics.register_metrics import RegisterMetrics
from lib.publisher.sns import SNSPublisher


class MetricNames:
    INCIDENT_DROPPED = "incident_dropped"
    INCIDENT_PUBLISHED = "incident_published"


class PrismPublisher:
    def __init__(self, sns_publisher: SNSPublisher, kvs_stream_arn: str):
        """
        Publisher for sending incident data to Prism to handle further processing,
        clip generation, and archival.


        Args:
            sns_publisher(SNSPublisher): publisher which makes the underlying calls
            kvs_stream_arn(str): Amazon resource name of the kinesis video stream from
                which all the incidents originate
        """
        self._sns_publisher = sns_publisher
        self._kvs_stream_arn = kvs_stream_arn

        atexit.register(self.close)

    @classmethod
    def _do_from_config_values(
        cls,
        camera_uuid: str,
        kvs_stream_arn: str,
        sns_topic_arn: str,
        metrics_client: RegisterMetrics,
        sns_publisher_cls: Type[SNSPublisher],
    ):
        def error_callback(err):
            logger.warning(f"Failed incident publish: {err}")
            if metrics_client:
                metrics_client.increment_metric_counter(
                    MetricNames.INCIDENT_DROPPED, 1
                )

        def success_callback(res):
            logger.debug(f"Successful incident publish: {res}")
            if metrics_client:
                metrics_client.increment_metric_counter(
                    MetricNames.INCIDENT_PUBLISHED, 1
                )

        sns_publisher = sns_publisher_cls(
            topic_arn=sns_topic_arn,
            message_group_id=camera_uuid,
            error_callback=error_callback,
            success_callback=success_callback,
        )

        return cls(
            sns_publisher=sns_publisher,
            kvs_stream_arn=kvs_stream_arn,
        )

    @classmethod
    def from_config_values(
        cls,
        camera_uuid: str,
        kvs_stream_arn: str,
        sns_topic_arn: str,
        metrics_client: RegisterMetrics = None,
    ):
        """
        Creates a PrismPublisher from values in the perception config

        Args:
            camera_uuid(str): the UUID of the camera from which incidents originate.
            kvs_stream_arn(str): the Amazon Resource Name of the Kinesis Video Stream from.
            sns_topic_arn(str): the Amazon Resource Name of the incident ingestion SNS topic to.
                publish to.
            metrics_client(Optional[RegisterMetrics]): client for open telemetry to track successful
                and failed incident publishes. Defaults to None for no tracking.

        Returns:
            PrismPublisher: the publisher
        """

        return cls._do_from_config_values(
            camera_uuid=camera_uuid,
            kvs_stream_arn=kvs_stream_arn,
            sns_topic_arn=sns_topic_arn,
            metrics_client=metrics_client,
            sns_publisher_cls=SNSPublisher,
        )

    def publish_incident(self, incident: Incident):
        """
        Sends incident to Prism for processing

        Args:
            incident(Incident): incident datum to put
        """

        incident_dict = incident.to_dict()
        incident_dict["kvs_stream_arn"] = self._kvs_stream_arn
        self._sns_publisher.put_message(
            message=json.dumps(incident_dict),
            deduplication_key=incident.uuid,
        )

    def publish_incidents(self, incidents: Iterable[Incident]):
        for incident in incidents:
            self.publish_incident(incident)

    def close(self):
        self._sns_publisher.close()
