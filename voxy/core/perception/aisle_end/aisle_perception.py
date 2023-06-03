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

from core.incidents.utils import CameraConfig
from core.structs.actor import (
    Actor,
    ActorCategory,
    get_actor_id_from_actor_category_and_track_id,
    get_track_uuid,
)
from core.structs.attributes import Point, Polygon


class AislePerception:
    def __init__(
        self,
        camera_uuid,
    ):
        self.camera_uuid = camera_uuid
        self.aisle_ends = None

    def __call__(self, frame):
        actors = []
        if self.aisle_ends is None:
            camera_config = CameraConfig(
                self.camera_uuid, frame.shape[0], frame.shape[1]
            )
            self.aisle_ends = camera_config.aisle_ends
        for track_id, aisle_end_line in enumerate(self.aisle_ends):
            actor_id = get_actor_id_from_actor_category_and_track_id(
                track_id, ActorCategory.AISLE_END
            )

            # add slight padding to line to make a polygon
            aisle_end_shapely = aisle_end_line.to_shapely_line()
            aisle_end_shifted_shapely = aisle_end_shapely.parallel_offset(
                1, "left"
            )
            aisle_end_polygon = Polygon(
                vertices=[
                    Point(
                        x=aisle_end_shapely.coords[0][0],
                        y=aisle_end_shapely.coords[0][1],
                    ),
                    Point(
                        x=aisle_end_shapely.coords[1][0],
                        y=aisle_end_shapely.coords[1][1],
                    ),
                    Point(
                        x=aisle_end_shifted_shapely.coords[1][0],
                        y=aisle_end_shifted_shapely.coords[1][1],
                    ),
                    Point(
                        x=aisle_end_shifted_shapely.coords[0][0],
                        y=aisle_end_shifted_shapely.coords[0][1],
                    ),
                ]
            )

            actors.append(
                Actor(
                    category=ActorCategory.AISLE_END,
                    track_id=actor_id,
                    polygon=aisle_end_polygon,
                    manual=False,
                    track_uuid=get_track_uuid(
                        camera_uuid=self.camera_uuid,
                        unique_identifier=str(actor_id),
                        category=ActorCategory.AISLE_END,
                    ),
                )
            )
        return actors
