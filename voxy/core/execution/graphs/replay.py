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
import typing

from loguru import logger

from core.execution.graphs.abstract import AbstractGraph
from core.execution.nodes.incident_writer import IncidentWriterNode
from core.execution.nodes.mega import MegaNode
from core.execution.nodes.publisher import PublisherNode
from core.structs.frame import Frame
from core.structs.incident import Incident


class ReplayGraph(AbstractGraph):
    """
    Utility graph to replay from kinesis datastream
    frame structs logged from production
    """

    def __init__(self, config: dict, frame_generator: typing.Iterator[Frame]):
        """
        Initializes the replay graph

        Args:
            config (dict): the config to generate the graph
            frame_generator (typing.Iterator[Frame]): frame generator for this execution
        """
        # This call modifies the config
        super().__init__(config)
        self._frame_generator = frame_generator
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

    def execute(self) -> typing.List[Incident]:
        """
        Executes the replay graph

        Returns:
            typing.List[Incident]: the list of incidents generated
                                   during the process
        """
        all_incidents = []
        frame_count = 0
        for camera_frame in self._frame_generator:
            frame_count += 1
            current_frame_struct = Frame.from_proto(camera_frame.frame)
            if self._incident_writer_node:
                self._incident_writer_node.process_frame(
                    None,
                    current_frame_struct,
                )

            incidents = self._mega_node.process(current_frame_struct)

            if incidents:
                if self._incident_writer_node:
                    self._incident_writer_node.process_incidents(incidents)
                all_incidents.extend(incidents)

        # Finalize all nodes
        finalized_incidents = self.finalize()
        all_incidents.extend(finalized_incidents)
        logger.info(
            f"Processed: {frame_count} frames, or about {frame_count/5/60} minutes at 5 FPS"
        )
        return all_incidents

    def finalize(self) -> typing.List[Incident]:
        """
        Finalize all the components of the replay graph

        Returns:
            typing.List[Incident]: any remaining incidents
                generated when finalizing the components of
                the develop graph
        """
        incidents = self._mega_node.finalize()
        if self._incident_writer_node:
            self._incident_writer_node.process_incidents(incidents)

            self._incident_writer_node.finalize()

            if self._publisher_node:
                self._publisher_node.finalize()
        return incidents
