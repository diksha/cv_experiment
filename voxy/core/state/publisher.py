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
import os
from concurrent import futures
from typing import Union

import boto3
from botocore.config import Config
from google import api_core
from google.cloud import pubsub
from google.cloud.pubsub import types
from google.protobuf.any_pb2 import Any

from core.state.utils import create_subscription, create_topic
from core.structs.actor import ActorCategory
from core.structs.event import Event
from core.structs.state import State
from lib.publisher.kinesis import KinesisPartitionPublisher


class Publisher:
    # trunk-ignore(pylint/R0913)
    def __init__(
        self,
        otlp_meter,
        batch_max_message,
        batch_max_latency_seconds,
        batch_max_bytes,
        retry_deadline_seconds,
        state_topic,
        event_topic,
        emulator_host,
        kinesis_data_stream=None,
        kinesis_partition_key=None,
    ):

        self._state_topic = state_topic
        self._event_topic = event_topic

        # This is for local development.
        # Ensure topic and subscriber exists in the emulator when running locally.
        if emulator_host is not None:
            os.environ["PUBSUB_EMULATOR_HOST"] = emulator_host
            create_topic(self._state_topic)
            create_subscription(
                self._state_topic,
                f"{self._state_topic.replace('topics', 'subscriptions')}-subscription",
            )
            create_topic(self._event_topic)
            create_subscription(
                self._event_topic,
                f"{self._event_topic.replace('topics', 'subscriptions')}-subscription",
            )

        if (
            kinesis_data_stream is not None
            and kinesis_partition_key is not None
            and otlp_meter is not None
        ):
            state_event_dropped = otlp_meter.create_counter(
                "state_event_dropped"
            )
            state_event_processed = otlp_meter.create_counter(
                "state_event_published"
            )

            def state_event_put_error_callback(_):
                state_event_dropped.add(1)

            def state_event_put_success_callback(_):
                state_event_processed.add(1)

            kinesis_config = Config(
                retries={"max_attempts": 10, "mode": "standard"}
            )

            self._kinesis_client = KinesisPartitionPublisher(
                stream_name=kinesis_data_stream,
                partition_key=kinesis_partition_key,
                error_callback=state_event_put_error_callback,
                success_callback=state_event_put_success_callback,
                kinesis_client=boto3.client("kinesis", config=kinesis_config),
                max_buffer_size=100000,
            )
        else:
            self._kinesis_client = None

        self._publish_client = pubsub.PublisherClient(
            batch_settings=types.BatchSettings(
                max_bytes=batch_max_bytes,
                max_latency=batch_max_latency_seconds,
                max_messages=batch_max_message,
            ),
            publisher_options=types.PublisherOptions(
                flow_control=types.PublishFlowControl(
                    message_limit=1000000,  # 1 Million Messages Buffer
                    byte_limit=1 * 1024 * 1024 * 1024,  # 1GB Memory Buffer
                    limit_exceeded_behavior=types.LimitExceededBehavior.IGNORE,
                ),
                retry=api_core.retry.Retry(
                    initial=1,  # seconds
                    maximum=600,  # seconds
                    multiplier=1.5,
                    deadline=retry_deadline_seconds,
                    predicate=api_core.retry.if_exception_type(
                        api_core.exceptions.Aborted,
                        api_core.exceptions.DeadlineExceeded,
                        api_core.exceptions.InternalServerError,
                        api_core.exceptions.ResourceExhausted,
                        api_core.exceptions.ServiceUnavailable,
                        api_core.exceptions.Unknown,
                        api_core.exceptions.Cancelled,
                    ),
                ),
            ),
        )

        self._is_production_env = (
            "voxel-production" in self._state_topic
            or "voxel-production" in self._event_topic
        )

        # This is only for development when using videos
        # So we can ensure all messages are published before
        # exiting.
        self._futures_timeout_seconds = 10
        self._futures = []

    @classmethod
    def should_generate_google_pubsub_message(
        cls, message: Union[State, Event]
    ) -> bool:
        """
        Returns whether to publish to the google pub sub system
        TODO: remove this when google pubsub has been removed

        Args:
            message (Union[State, Event]): the raw input message to analyze

        Returns:
            bool: whether to publish to google pub sub
        """
        if not isinstance(message, State):
            return False

        return message.actor_category == ActorCategory.MOTION_DETECTION_ZONE

    def publish(self, message: Union[State, Event]):
        """Publish message to GCP pubsub and Kinesis (if enabled)

        Args:
            message (Union[State, Event]): data to publish

        Raises:
            TypeError: on incorrect message type
        """
        if isinstance(message, State):
            topic = self._state_topic
        elif isinstance(message, Event):
            topic = self._event_topic
        else:
            raise TypeError(f"Unknown message type: {type(message)}")

        messagepb = message.to_proto()
        future = None

        # all states for production line down are the only ones we need to publish
        if self.should_generate_google_pubsub_message(message):
            # send the message to google pubsub
            future = self._publish_client.publish(
                topic, messagepb.SerializeToString()
            )

        # send the message to kinesis if it is configured
        if self._kinesis_client is not None:
            wrapped = Any()
            wrapped.Pack(messagepb)
            self._kinesis_client.put_datum(
                base64.b64encode(wrapped.SerializeToString())
            )

        if not self._is_production_env and future is not None:
            self._futures.append(future)

    def finalize(self):
        futures.wait(
            self._futures,
            timeout=self._futures_timeout_seconds,
            return_when=futures.ALL_COMPLETED,
        )
