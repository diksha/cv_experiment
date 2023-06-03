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

from datetime import datetime, timezone
from uuid import uuid4

from neomodel import (
    ArrayProperty,
    BooleanProperty,
    DateTimeProperty,
    IntegerProperty,
    JSONProperty,
    RelationshipTo,
    StringProperty,
    StructuredNode,
)

from core.metaverse.models.model import (  # trunk-ignore(flake8/F401,pylint/W0611)
    Task,
)


class Camera(StructuredNode):
    __primarykey__ = "uuid"
    uuid = StringProperty(unique_index=True, default=uuid4)
    created_timestamp = DateTimeProperty(
        default=lambda: datetime.now(timezone.utc)
    )
    organization = StringProperty()
    location = StringProperty()
    zone = StringProperty()
    channel_name = StringProperty()
    kinesis_url = StringProperty()
    is_active = BooleanProperty()
    camera_config_ref = RelationshipTo("CameraConfig", "CameraConfig")
    task_ref = RelationshipTo("Task", "Task")


class CameraConfig(StructuredNode):
    __primarykey__ = "uuid"
    uuid = StringProperty(unique_index=True, default=uuid4)
    version = IntegerProperty()
    doors = ArrayProperty(JSONProperty())
    driving_areas = ArrayProperty(JSONProperty())
    actionable_regions = ArrayProperty(JSONProperty())
    intersections = ArrayProperty(JSONProperty())
    end_of_aisles = ArrayProperty(JSONProperty())
    no_pedestrian_zones = ArrayProperty(JSONProperty())
    motion_detection_zones = ArrayProperty(JSONProperty())
    no_obstruction_regions = ArrayProperty(JSONProperty())
