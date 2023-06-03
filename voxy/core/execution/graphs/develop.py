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

import threading

from loguru import logger

from core.execution.graphs.abstract import AbstractGraph
from core.execution.nodes.annotation import AnnotationNode
from core.execution.nodes.incident_writer import IncidentWriterNode
from core.execution.nodes.mega import MegaNode
from core.execution.nodes.perception import PerceptionNode
from core.execution.nodes.publisher import PublisherNode
from core.execution.nodes.video_stream import VideoStreamNode
from core.execution.nodes.video_writer import VideoWriterNode
from core.structs.frame import StampedImage
from core.utils.aws_utils import does_s3_blob_exists


class DevelopGraph(AbstractGraph):
    def __init__(self, config, perception_runner_context):
        # This call modifies the config
        super().__init__(config)
        self._stream_node = VideoStreamNode(config)
        # daemon=True ensures that thread exits when main process exits
        threading.Thread(target=self._stream_node.run, daemon=True).start()

        config["camera"] = config.get("camera", {})
        config["camera"]["width"] = self._stream_node.get_width()
        config["camera"]["height"] = self._stream_node.get_height()
        config["camera"]["frame_rate"] = self._stream_node.get_frame_rate()

        self._annotation_node = AnnotationNode(config)

        if not self._annotation_node.is_cached():
            self._perception_node = PerceptionNode(
                config=config,
                perception_runner_context=perception_runner_context,
            )
        else:
            logger.debug("Perception already cached")
            self._perception_node = None

        self.video_uuid = config["video_uuid"]
        cache_key = config["cache_key"]
        s3_video_output_path = (
            f"s3://voxel-temp/video_cache/{cache_key}/{self.video_uuid}.mp4"
        )
        config["video_writer"] = config.get("video_writer", {})
        config["video_writer"]["output_s3_path"] = s3_video_output_path

        # Add video writer if enabled
        if config["enable_video_writer"] and not does_s3_blob_exists(
            s3_video_output_path
        ):
            self._video_writer_node = VideoWriterNode(config)
        else:
            self._video_writer_node = None

        # Initialize Mega Node
        self._mega_node = MegaNode(config)
        if not config["incident"]["dry_run"]:
            self._incident_writer_node = IncidentWriterNode(config)
            # daemon=True ensures that thread exits when main process exits
            threading.Thread(
                target=self._incident_writer_node.run, daemon=True
            ).start()

            if config.get("publisher", {}).get("enabled"):
                config["incident"] = config.get("incident", {})
                config["incident"][
                    "temp_directory"
                ] = self._incident_writer_node.get_incident_temp_directory()
                self._publisher_node = PublisherNode(config)
                # daemon=True ensures that thread exits when main process exits
                threading.Thread(
                    target=self._publisher_node.run, daemon=True
                ).start()
            else:
                self._publisher_node = None
        else:
            self._incident_writer_node = None

    def execute(self):
        logger.info(f"Processing {self.video_uuid}")

        all_incidents = []
        while True:
            frame, frame_ms = self._stream_node.get_next_frame()
            if frame is None and self._stream_node.is_video_complete():
                break

            if frame is None:
                continue

            if self._video_writer_node is not None:
                self._video_writer_node.process_frame(frame, frame_ms)

            if self._perception_node is not None:
                current_frame_struct = self._perception_node.process(
                    StampedImage(image=frame, timestamp_ms=frame_ms)
                )
                self._annotation_node.cache_pred_annotation(
                    current_frame_struct
                )
            else:
                current_frame_struct = (
                    self._annotation_node.get_pred_annotation(frame_ms)
                )
            if self._incident_writer_node:
                self._incident_writer_node.process_frame(
                    frame,
                    current_frame_struct,
                )

            if self._video_writer_node is not None:
                gt_frame_struct = self._annotation_node.get_gt_annotation(
                    frame_ms
                )
                self._video_writer_node.process_frame_struct(
                    current_frame_struct, gt_frame_struct
                )

            incidents = self._mega_node.process(current_frame_struct)

            if incidents:
                if self._incident_writer_node:
                    self._incident_writer_node.process_incidents(incidents)
                all_incidents.extend(incidents)

        # Finalize all nodes
        finalized_incidents = self.finalize()
        all_incidents.extend(finalized_incidents)
        logger.info(f"Incidents: {all_incidents}")
        return all_incidents

    def finalize(self):
        if self._perception_node:
            self._perception_node.finalize()

        if self._annotation_node:
            self._annotation_node.finalize()

        if self._video_writer_node:
            self._video_writer_node.finalize()

        incidents = self._mega_node.finalize()
        if self._incident_writer_node:
            self._incident_writer_node.process_incidents(incidents)

            self._incident_writer_node.finalize()

            if self._publisher_node:
                self._publisher_node.finalize()

        return incidents
