#
# Copyright 2020-2021 Voxel Labs, Inc.
# All rights reserved.
#
# This document may not be reproduced, republished, distributed, transmitted,
# displayed, broadcast or otherwise exploited in any manner without the express
# prior written permission of Voxel Labs, Inc. The receipt or possession of this
# document does not convey any rights to reproduce, disclose, or distribute its
# contents, or to manufacture, use, or sell anything that it may describe, in
# whole or in part.
#

import base64
import threading
from typing import Iterable, Optional

import boto3
from botocore.config import Config
from loguru import logger

from core.execution.graphs.abstract import AbstractGraph
from core.execution.graphs.metrics_constants import (
    AttributeNames,
    MetricNames,
    SpanNames,
)
from core.execution.nodes.camera_stream import CameraStreamNode
from core.execution.nodes.incident_writer import IncidentWriterNode
from core.execution.nodes.mega import MegaNode
from core.execution.nodes.perception import PerceptionNode
from core.execution.nodes.publisher import PublisherNode
from core.structs.frame import StampedImage
from core.structs.incident import Incident
from lib.infra.metrics.constants import Environment
from lib.infra.metrics.register_metrics import RegisterMetrics
from lib.publisher.kinesis import KinesisPartitionPublisher
from lib.utils.aws.session import (
    assume_role_for_default_session,
    create_session_with_role,
)

# trunk-ignore(pylint/E0611)
from protos.perception.structs.v1.frame_pb2 import CameraFrame
from services.platform.prism.lib.publisher import PrismPublisher

DEFAULT_KINESIS_DATA_STREAM = "perception-states-events"
DEFAULT_KINESIS_FRAME_STRUCTS_STREAM = "perception-frame-structs"


class ProductionGraph(AbstractGraph):
    def __init__(self, config, env, perception_runner_context):
        # This call modifies the config
        super().__init__(config)
        self._config = config
        self.metrics = RegisterMetrics(
            service_name="production_graph",
            metric_names=MetricNames.get_all_metric_names(),
            environment=env,
            attributes={
                "camera_uuid": self._config["camera_uuid"],
            },
        )

        self._kvs_read_session = self._assume_kvs_read_role()
        self._assume_default_iam_role()

        self._kinesis_frame_structs = self._create_frame_struct_publisher()
        self._prism_client = self._create_incident_publisher()

        self._no_frame_sleep_duration_seconds = 0.1

        self._stream_node = CameraStreamNode(config, self._kvs_read_session)
        self._stream_node.start()

        config["camera"] = config.get("camera", {})

        # set default kinesis data stream
        config.setdefault("state", {})
        config["state"].setdefault("publisher", {})
        config["state"]["publisher"].setdefault(
            "kinesis_data_stream", DEFAULT_KINESIS_DATA_STREAM
        )
        config["state"]["publisher"]["kinesis_partition_key"] = config.get(
            "camera_uuid"
        )

        self._perception_node = PerceptionNode(
            config=config, perception_runner_context=perception_runner_context
        )
        self._mega_node = MegaNode(config, self.metrics.get_metrics_meter())

        self._incident_writer_node = IncidentWriterNode(
            config, self._kvs_read_session
        )
        threading.Thread(
            target=self._incident_writer_node.run, daemon=True
        ).start()

        config["incident"] = config.get("incident", {})
        config["incident"][
            "temp_directory"
        ] = self._incident_writer_node.get_incident_temp_directory()
        self._publisher_node = PublisherNode(config)
        threading.Thread(target=self._publisher_node.run, daemon=True).start()

    def _get_environment(self, environment: str) -> Environment:
        if environment == "production":
            return Environment.PRODUCTION
        if environment == "staging":
            return Environment.STAGING
        return Environment.DEVELOPMENT

    def _assume_kvs_read_role(self) -> boto3.Session:
        """Assume an IAM role for reading from KinesisVideoStreams
        Used for development runs of ProductionGraph

        Returns:
            boto3.Session: session with the KVS read role
        """
        read_kvs_iam_role_arn = self._config.get("assume_role", {}).get(
            "read_kvs_arn"
        )

        if read_kvs_iam_role_arn is not None:
            logger.info("Assuming IAM role for reading KinesisVideoStreams")
            return create_session_with_role(read_kvs_iam_role_arn, "read_kvs")

        return None

    def _assume_default_iam_role(self):
        """Override the default boto3 session with an assumed IAM role
        Used for development runs of ProductionGraph
        """
        default_iam_role_arn = self._config.get("assume_role", {}).get(
            "default_role_arn"
        )
        if default_iam_role_arn is not None:
            logger.info(
                "Overriding default boto3 session with assumed IAM role"
            )
            assume_role_for_default_session(default_iam_role_arn)

    def _create_frame_struct_publisher(
        self,
    ) -> Optional[KinesisPartitionPublisher]:
        enabled = self._config.get("publisher", {}).get("enabled", False)

        if not enabled:
            return None

        kinesis_frame_structs_stream = self._config.get("publisher", {}).get(
            "kinesis_frame_structs_stream",
            DEFAULT_KINESIS_FRAME_STRUCTS_STREAM,
        )

        kinesis_config = Config(
            retries={"max_attempts": 10, "mode": "standard"}
        )

        def frame_struct_put_error_callback(_):

            self.metrics.increment_metric_counter(
                MetricNames.FRAME_STRUCTS_DROPPED,
                1,
            )

        def frame_struct_put_success_callback(_):
            self.metrics.increment_metric_counter(
                MetricNames.FRAME_STRUCTS_PUBLISHED,
                1,
            )

        return KinesisPartitionPublisher(
            stream_name=kinesis_frame_structs_stream,
            partition_key=self._config.get("camera_uuid"),
            error_callback=frame_struct_put_error_callback,
            success_callback=frame_struct_put_success_callback,
            kinesis_client=boto3.client("kinesis", config=kinesis_config),
        )

    def _publish_frame_struct(self, frame_struct):
        if self._kinesis_frame_structs is not None:
            self._kinesis_frame_structs.put_datum(
                base64.b64encode(
                    CameraFrame(
                        camera_uuid=self._config["camera_uuid"],
                        run_uuid=self._config["run_uuid"],
                        frame=frame_struct.to_proto(),
                    ).SerializeToString()
                )
            )

    def _create_incident_publisher(self) -> Optional[PrismPublisher]:
        """Create a publisher to transfer incident data to Prism"""

        enabled = (
            self._config.get("incident", {})
            .get("publisher", {})
            .get("enabled", False)
        )

        if not enabled:
            return None

        return PrismPublisher.from_config_values(
            camera_uuid=self._config["camera_uuid"],
            kvs_stream_arn=self._config["camera"]["arn"],
            sns_topic_arn=self._config["incident"]["publisher"][
                "sns_topic_arn"
            ],
            metrics_client=self.metrics,
        )

    def _publish_incidents(self, incidents: Iterable[Incident]):
        """Sends incident data to Prism for processing if publisher is defined"""
        if self._prism_client is None:
            return

        logger.debug(f"Publishing {len(incidents)} incidents")
        self._prism_client.publish_incidents(incidents)

    # trunk-ignore(pylint/C0116)
    def execute(self):
        last_total_number_of_frames_dropped = 0
        while True:
            with self.metrics.get_metrics_tracer().start_as_current_span(
                SpanNames.EXECUTE, attributes={"name": AttributeNames.EXECUTE}
            ) as execute_span:
                with self.metrics.get_metrics_tracer().start_as_current_span(
                    SpanNames.STREAM_NODE
                ):
                    total_number_of_frames_dropped = (
                        self._stream_node.get_total_number_of_frames_dropped()
                    )
                    self.metrics.increment_metric_counter(
                        MetricNames.FRAMES_DROPPED,
                        total_number_of_frames_dropped
                        - last_total_number_of_frames_dropped,
                    )
                    last_total_number_of_frames_dropped = (
                        total_number_of_frames_dropped
                    )

                    frame, frame_ms = self._stream_node.get_next_frame()
                    execute_span.set_attribute(
                        AttributeNames.FRAME_MS, frame_ms
                    )

                    if frame is None:
                        continue

                logger.info(f"Processing {frame_ms}")

                with self.metrics.get_metrics_tracer().start_as_current_span(
                    SpanNames.PERCEPTION_NODE
                ):
                    current_frame_struct = self._perception_node.process(
                        StampedImage(image=frame, timestamp_ms=frame_ms)
                    )

                self._publish_frame_struct(current_frame_struct)

                with self.metrics.get_metrics_tracer().start_as_current_span(
                    SpanNames.MEGA_NODE
                ):
                    incidents = self._mega_node.process(current_frame_struct)

                self._publish_incidents(incidents)

                with self.metrics.get_metrics_tracer().start_as_current_span(
                    SpanNames.INCIDENT_WRITER_NODE
                ):
                    # TODO(diksha): Remove the tuple and only send frame struct.
                    self._incident_writer_node.process_frame(
                        None,
                        current_frame_struct,
                    )
                    self._incident_writer_node.process_incidents(incidents)

                self.metrics.increment_metric_counter(
                    MetricNames.FRAMES_PROCESSED,
                    1,
                )
