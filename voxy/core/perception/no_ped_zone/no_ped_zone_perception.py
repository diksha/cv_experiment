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


class NoPedZonePerception:
    def __init__(
        self,
        camera_uuid,
    ):
        self.camera_uuid = camera_uuid
        self.no_pedestrian_zones = None

    def __call__(self, frame):
        actors = []
        if self.no_pedestrian_zones is None:
            camera_config = CameraConfig(
                self.camera_uuid, frame.shape[0], frame.shape[1]
            )
            self.no_pedestrian_zones = camera_config.no_pedestrian_zones
        for track_id, zone_polygon in enumerate(self.no_pedestrian_zones):
            actor_id = get_actor_id_from_actor_category_and_track_id(
                track_id, ActorCategory.NO_PED_ZONE
            )
            actors.append(
                Actor(
                    category=ActorCategory.NO_PED_ZONE,
                    track_id=actor_id,
                    polygon=zone_polygon,
                    manual=False,
                    track_uuid=get_track_uuid(
                        camera_uuid=self.camera_uuid,
                        unique_identifier=str(actor_id),
                        category=ActorCategory.NO_PED_ZONE,
                    ),
                )
            )
        return actors
