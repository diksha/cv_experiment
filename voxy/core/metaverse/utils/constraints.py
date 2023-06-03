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
from neomodel import db

from core.metaverse.metaverse import Metaverse


def apply_constraints():
    Metaverse("INTERNAL")
    camera_constraint = "CREATE CONSTRAINT camera_constraint IF NOT EXISTS \
            FOR (n:Camera) REQUIRE (n.organization, n.location, n.zone, n.channel_name) IS NODE KEY "
    logset_constraint = "CREATE CONSTRAINT logset_name IF NOT EXISTS FOR (logset:Logset) REQUIRE logset.name IS UNIQUE"
    voxel_uuid_constraint = "CREATE CONSTRAINT voxel_uuid_constraint IF NOT EXISTS FOR (video:Video) REQUIRE video.voxel_uuid IS UNIQUE"
    labeling_tool_constraint = (
        "CREATE CONSTRAINT labeling_tool_constraint IF NOT EXISTS FOR"
        " (labeling_tool:LabelingTool) REQUIRE labeling_tool.name IS UNIQUE"
    )
    constraints = [
        camera_constraint,
        logset_constraint,
        voxel_uuid_constraint,
        labeling_tool_constraint,
    ]
    for constraint in constraints:
        db.cypher_query(constraint)


if __name__ == "__main__":
    apply_constraints()
