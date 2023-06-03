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
import os
import time
from pathlib import Path

from loguru import logger

from core.execution.nodes.abstract import AbstractNode
from core.execution.utils.graph_config_utils import (
    get_should_generate_cooldown_incidents_from_config,
)
from core.utils.incident_writer import IncidentWriter, IncidentWriterInput


class IncidentWriterNode(AbstractNode):
    def __init__(self, config, kvs_read_session=None):
        if config["incident"]["generate_temp_subdirs"]:
            config["incident"]["temp_directory"] = os.path.join(
                config["incident"]["temp_directory"], str(time.time())
            )
        self._should_generate_cooldown_incidents = (
            get_should_generate_cooldown_incidents_from_config(config)
        )
        self._incident_temp_directory = config["incident"]["temp_directory"]
        self._dry_run = config["incident"]["dry_run"]
        self._camera_config_version = config["camera"].get("version", 1)
        number_of_pre_buffer_post_buffer_frames = 1000
        self._controller = IncidentWriter(
            input_params=IncidentWriterInput(
                temp_directory=self._incident_temp_directory,
                dry_run=self._dry_run,
                camera_arn=config.get("camera", {}).get("arn", None),
                video_uuid=config.get("video_uuid", None),
                fps=config.get("camera", {}).get("fps", None),
                should_generate_cooldown_incidents=self._should_generate_cooldown_incidents,
                kvs_read_session=kvs_read_session,
                frame_struct_buffer_size=int(
                    config["state"]["frames_to_buffer"]
                )
                + number_of_pre_buffer_post_buffer_frames,
            )
        )
        Path(self._incident_temp_directory).mkdir(parents=True, exist_ok=True)

        self._finalized = False

    def get_incident_temp_directory(self):
        return self._incident_temp_directory

    def process_frame(self, frame, frame_struct):
        self._controller.insert_frame_and_struct(
            frame_struct.relative_timestamp_ms,
            (
                frame,
                frame_struct,
            ),
        )

    def process_incidents(self, incidents):
        for incident in incidents:
            incident.camera_config_version = self._camera_config_version
        self._controller.add_incident(incidents)

    def run(self):
        # Keep processing incident queue in the background to write incidents to disk.
        while not self._finalized:
            try:
                # This used to sleep but it should no longer need to do so as the implementation is now based
                # on a queue, which will block using proper synchronization primitives without busy waiting
                self._controller.process_next()
            # trunk-ignore(pylint/W0703)
            except Exception as e:
                # In case of exception force the parent actor
                # to crash using ray.actor.exit_actor()
                # Otherwise it will stay stuck here silently.
                logger.exception(f"Incident Writer Node Run: {e}")
                # trunk-ignore(pylint/W0212)
                os._exit(1)

    def finalize(self):
        self._controller.finalize()
        self._finalized = True
