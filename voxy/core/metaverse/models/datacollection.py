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
    FloatProperty,
    IntegerProperty,
    JSONProperty,
    One,
    RelationshipTo,
    StringProperty,
    StructuredNode,
)

from core.metaverse.models.camera import (  # noqa pylint: disable=unused-import
    Camera,
)


class DataCollection(StructuredNode):
    """Collection of data(images or videos)

    Args:
        StructuredNode: neomodel Node type
    """

    __primarykey__ = "uuid"
    uuid = StringProperty(unique_index=True, default=uuid4)
    created_timestamp = DateTimeProperty(
        default=lambda: datetime.now(timezone.utc)
    )
    camera_ref = RelationshipTo("Camera", "Camera")
    name = StringProperty()
    path = StringProperty()
    data_collection_type = StringProperty()
    voxel_uuid = StringProperty(unique_index=True)
    is_test = BooleanProperty()
    is_deprecated = BooleanProperty()
    frame_ref = RelationshipTo("Frame", "Frame")
    label_metadata_ref = RelationshipTo(
        "LabelMetadata", "LabelMetadata", cardinality=One
    )


class LabelMetadata(StructuredNode):
    """Labeling metadata for a video

    Args:
        StructuredNode: neomodel Node type
    """

    __primarykey__ = "uuid"
    uuid = StringProperty(unique_index=True, default=uuid4)
    source = StringProperty()
    project_name = StringProperty()
    taxonomy = StringProperty()
    taxonomy_version = ArrayProperty(StringProperty())


class VersionedViolation(StructuredNode):
    __primarykey__ = "uuid"
    uuid = StringProperty(unique_index=True, default=uuid4)
    version = StringProperty()
    violations = ArrayProperty(StringProperty())


class Frame(StructuredNode):
    __primarykey__ = "uuid"
    uuid = StringProperty(unique_index=True, default=uuid4)
    frame_number = IntegerProperty()
    frame_width = IntegerProperty()
    frame_height = IntegerProperty()
    relative_timestamp_s = FloatProperty()
    relative_timestamp_ms = FloatProperty()
    epoch_timestamp_ms = FloatProperty()
    path_of_image = StringProperty()
    actors_ref = RelationshipTo("Actor", "Actor")
    relative_image_path = StringProperty()


class Actor(StructuredNode):
    __primarykey__ = "uuid"
    uuid = StringProperty(unique_index=True, default=uuid4)
    category = StringProperty()
    occluded = BooleanProperty()
    occluded_degree = StringProperty(default="NONE")
    manual = BooleanProperty()
    mirrored = BooleanProperty()
    truncated = BooleanProperty()
    human_operating = BooleanProperty()
    forklift = BooleanProperty()
    loaded = BooleanProperty()
    forks_raised = BooleanProperty()
    operating_pit = BooleanProperty()
    operating_object = StringProperty()
    is_wearing_hard_hat = BooleanProperty()
    motion_detection_zone_state = StringProperty()
    is_wearing_safety_vest = BooleanProperty()
    is_wearing_safety_vest_v2 = BooleanProperty()
    is_carrying_object = BooleanProperty()
    door_state = StringProperty()
    door_type = StringProperty()
    track_id = IntegerProperty()
    track_uuid = StringProperty()
    polygon = JSONProperty()
    pose = StringProperty()
    activity = JSONProperty()
    head_covering_type = StringProperty()
