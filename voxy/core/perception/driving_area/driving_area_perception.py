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


class DrivingAreaPerception:
    def __init__(
        self,
        camera_uuid,
    ):
        self.camera_uuid = camera_uuid
        self.driving_areas = None

    def __call__(self, frame):
        if self.driving_areas is None:
            camera_config = CameraConfig(
                self.camera_uuid, frame.shape[0], frame.shape[1]
            )
            self.driving_areas = camera_config.driving_areas
        actors = []
        for track_id, zone_polygon in enumerate(self.driving_areas):
            actor_id = get_actor_id_from_actor_category_and_track_id(
                track_id, ActorCategory.DRIVING_AREA
            )
            actors.append(
                Actor(
                    category=ActorCategory.DRIVING_AREA,
                    track_id=actor_id,
                    polygon=zone_polygon,
                    manual=False,
                    track_uuid=get_track_uuid(
                        camera_uuid=self.camera_uuid,
                        unique_identifier=str(actor_id),
                        category=ActorCategory.DRIVING_AREA,
                    ),
                )
            )
        return actors
