#
# Copyright 2023 Voxel Labs, Inc.
# All rights reserved.
#
# This document may not be reproduced, republished, distributed, transmitted,
# displayed, broadcast or otherwise exploited in any manner without the express
# prior written permission of Voxel Labs, Inc. The receipt or possession of this
# document does not convey any rights to reproduce, disclose, or distribute its
# contents, or to manufacture, use, or sell anything that it may describe, in
# whole or in part.
#


class SpanNames:
    STREAM_NODE = "stream_node"
    PERCEPTION_NODE = "perception_node"
    MEGA_NODE = "mega_node"
    INCIDENT_WRITER_NODE = "incident_writer_node"
    EXECUTE = "execute"


class AttributeNames:
    FRAME_MS = "frame_ms"
    EXECUTE = "execute"


class MetricNames:
    FRAMES_DROPPED = "frames_dropped"
    FRAMES_PROCESSED = "frames_processed"
    FRAME_STRUCTS_DROPPED = "frame_structs_dropped"
    FRAME_STRUCTS_PUBLISHED = "frame_structs_published"

    @staticmethod
    def get_all_metric_names() -> [str]:
        """
        Returns a list of all metrics defined in this class

        Returns:
            list: returns of all metric names defined in this class
        """

        return [
            value
            for name, value in vars(MetricNames).items()
            if not callable(value)
            and not name.startswith("__")
            and isinstance(value, str)
        ]
