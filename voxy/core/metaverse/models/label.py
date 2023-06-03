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
from datetime import datetime, timezone
from uuid import uuid4

from neomodel import (
    ArrayProperty,
    DateTimeProperty,
    JSONProperty,
    RelationshipTo,
    StringProperty,
    StructuredNode,
)


class LabelingTool(StructuredNode):
    """Node describing a labeling tool

    Args:
        StructuredNode: parent class
    """

    __primarykey__ = "uuid"
    uuid = StringProperty(unique_index=True, default=uuid4)
    created_timestamp = DateTimeProperty(
        default=lambda: datetime.now(timezone.utc)
    )
    name = StringProperty(unique_index=True)
    project_ref = RelationshipTo("LabelProject", "LabelProject")


class LabelProject(StructuredNode):
    """Node describing a labeling project

    Args:
        StructuredNode: parent class
    """

    __primarykey__ = "uuid"
    uuid = StringProperty(unique_index=True, default=uuid4)
    created_timestamp = DateTimeProperty(
        default=lambda: datetime.now(timezone.utc)
    )
    name = StringProperty()
    description = StringProperty()
    taxonomy_ref = RelationshipTo("Taxonomy", "Taxonomy")
    last_checked_timestamp = DateTimeProperty()


class Taxonomy(StructuredNode):
    """Node describing a taxonomy version

    Args:
        StructuredNode: parent class
    """

    __primarykey__ = "uuid"
    uuid = StringProperty(unique_index=True, default=uuid4)
    created_timestamp = DateTimeProperty(
        default=lambda: datetime.now(timezone.utc)
    )
    taxonomy_version = StringProperty()
    taxonomy_json = JSONProperty()
    tags = ArrayProperty(StringProperty())
    notes = ArrayProperty(StringProperty())
