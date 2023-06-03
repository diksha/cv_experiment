#
# Copyright 2023 Voxel Labs, Inc.
# All rights reserved.

# This document may not be reproduced, republished, distributed, transmitted,
# displayed, broadcast or otherwise exploited in any manner without the express
# prior written permission of Voxel Labs, Inc. The receipt or possession of this
# document does not convey any rights to reproduce, disclose, or distribute its
# contents, or to manufacture, use, or sell anything that it may describe, in
# whole or in part.
#

import typing

import numpy as np
from loguru import logger

from core.perception.acausal.algorithms.base import BaseAcausalAlgorithm
from core.perception.acausal.algorithms.proximity.pit_person_proximity import (
    PitPersonProximityAlgorithm,
)
from core.perception.acausal.algorithms.proximity.pit_pit_proximity import (
    PitPitProximityAlgorithm,
)
from core.structs.actor import ActorCategory
from core.structs.tracklet import Tracklet
from core.structs.vignette import Vignette


class ProximityAlgorithmController(BaseAcausalAlgorithm):
    def __init__(self, config: dict) -> None:
        self._proximity_algorithms = [
            PitPitProximityAlgorithm(),
            PitPersonProximityAlgorithm(),
        ]
        self._person_proximity_threshold = (
            config.get("incident", {})
            .get("incident_machine_params", {})
            .get("pit_near_miss", {})
            .get("person_proximity_threshold", 0.02)
        )
        self._pit_proximity_threshold = (
            config.get("incident", {})
            .get("incident_machine_params", {})
            .get("pit_near_miss", {})
            .get("pit_proximity_threshold", 0.08)
        )

    def _update_proximity_metadata(
        self,
        actor_category: ActorCategory,
        proximal_actors_dict: typing.Dict[int, float],
        tracklet: Tracklet,
    ):
        """Update the boolean checks for pit or person proximity if any actor is within threshold

        Args:
            actor_category (ActorCategory): PERSON or PIT actor category
            proximal_actors_dict (dict): A dictionary of proximities of each actor to tracklet actor
            tracklet (Tracklet): Pit tracklet of interest
        """
        nearest_actor_proximity = min(
            np.array(list(proximal_actors_dict.values()))
        )
        if actor_category == ActorCategory.PERSON:
            tracklet.nearest_person_pixel_proximity = nearest_actor_proximity
            if nearest_actor_proximity < self._person_proximity_threshold:
                tracklet.is_proximal_to_person = True
            else:
                tracklet.is_proximal_to_person = False
        elif actor_category == ActorCategory.PIT:
            tracklet.nearest_pit_pixel_proximity = nearest_actor_proximity
            if nearest_actor_proximity < self._pit_proximity_threshold:
                tracklet.is_proximal_to_pit = True
            else:
                tracklet.is_proximal_to_pit = False

    def process_vignette(self, vignette: Vignette) -> Vignette:
        """Updates the tracklets with the proximity map and corresponding
            attributes

        Args:
            vignette (Vignette): Vignette to update proximities

        Returns:
            Vignette: Vignette with updated proximity data in tracklets
        """
        frame_timestamp_ms = vignette.present_timestamp_ms
        logger.debug("Processing vignette in proximity controller")
        for algorithm in self._proximity_algorithms:
            all_pit_proximity_dicts = algorithm.find_proximity(vignette)
            proximal_actors_category = algorithm.actor_categories[1]
            for (
                ego_pit_track_id,
                proximal_actors_dict,
            ) in all_pit_proximity_dicts.items():
                ego_pit_tracklet = vignette.tracklets[ego_pit_track_id]
                ego_pit_tracklet.update_tracklet_proximity(
                    proximal_actors_dict,
                    frame_timestamp_ms,
                    proximal_actors_category,
                )
                self._update_proximity_metadata(
                    actor_category=proximal_actors_category,
                    proximal_actors_dict=proximal_actors_dict,
                    tracklet=ego_pit_tracklet,
                )

        return vignette
