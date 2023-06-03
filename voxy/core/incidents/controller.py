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

from collections import deque
from typing import List

from core.incidents.utils import iter_monitors


class IncidentController:
    """Coordinates incident monitors and publishes findings downstream."""

    def __init__(
        self,
        camera_uuid: str = "",
        dry_run: bool = False,
        monitors_requested: List[str] = ["all"],
        hot_reload: bool = False,
        **kwargs,
    ):
        del kwargs  # unused keyword arguments
        self._camera_uuid = camera_uuid
        self._dry_run = dry_run
        self._monitors_requested = monitors_requested
        self._hot_reload = hot_reload
        self._default_thumbnail_gcs_path = None
        self._incident_queue = deque([])
        self._last_frame_time_ms = 0

        self._initialize_monitors_to_run(self._monitors_requested)

    def _initialize_monitors_to_run(self, monitors_requested):
        self._monitors = []
        for monitor in iter_monitors(
            monitors_requested=monitors_requested, hot_reload=self._hot_reload
        ):
            self._monitors.append(monitor(self._camera_uuid))
            print(f"Initialized {monitor.NAME} monitor")

    def process(self, frame_struct):
        if frame_struct is None:
            return []
        incidents = []
        for monitor in self._monitors:
            incidents += monitor.process_frame(frame_struct)

        self._last_frame_time_ms = frame_struct.relative_timestamp_ms

        if not self._dry_run:
            self._incident_queue.extend(incidents)
        else:
            if len(incidents) > 0:
                # TODO: pretty print incident details
                print(
                    "Found",
                    str(len(incidents)),
                    "incidents. Did not write to disk as dry_run is enabled.",
                )
        return incidents
