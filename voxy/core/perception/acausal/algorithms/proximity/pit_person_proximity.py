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

from core.perception.acausal.algorithms.common.distance_utils import (
    get_proximity_area_ratio,
)
from core.perception.acausal.algorithms.proximity.base_proximity_algorithm import (
    BaseProximityAlgorithm,
)
from core.structs.actor import ActorCategory
from core.structs.tracklet import Tracklet
from core.structs.vignette import Vignette


class PitPersonProximityAlgorithm(BaseProximityAlgorithm):
    def __init__(self) -> None:
        self.actor_categories = [ActorCategory.PIT, ActorCategory.PERSON]

    @staticmethod
    def _check_if_person_associated(tracklet: Tracklet):
        """Check if the person is associated with any PIT"""
        return tracklet.is_associated_with_pit

    @staticmethod
    def _get_pit_proximity_dict(
        proximity_area_ratio: np.ndarray,
        valid_pit_track_ids: np.ndarray,
        valid_person_track_ids: np.ndarray,
        tracklets: typing.Dict[int, Tracklet],
        frame_timestamp_ms: int,
    ):
        """Get the proximity dictionary of PIT and PERSON actors

        Args:
            proximity_area_ratio (np.ndarray): rows (PIT), columns (PERSON)
            valid_pit_track_ids (np.ndarray): An array of valid PIT track ids
            valid_person_track_ids (np.ndarray): A array of valid PERSON track ids
            tracklets (dict[int,Tracklet]): A dictionary of tracklets
            frame_timestamp_ms (int): A timestamp in milliseconds

        Returns:
            dict: A dictionary of the form {pit_track_id: {person_track_id: proximity_in pixels}}
        """
        proximity_area_dict = {}
        for pit_index, pit_track_id in enumerate(valid_pit_track_ids):
            for person_index, person_track_id in enumerate(
                valid_person_track_ids
            ):
                if not PitPersonProximityAlgorithm._check_if_person_associated(
                    tracklet=tracklets[person_track_id],
                ):
                    proximity_area_dict[pit_track_id] = {
                        person_track_id: proximity_area_ratio[
                            pit_index, person_index
                        ]
                    }
        return proximity_area_dict

    def find_proximity(self, vignette: Vignette) -> typing.Dict[int, dict]:
        """Find the proximity between actors in given vignette

        Args:
            vignette (Vignette): The vignette at the current frame

        Returns:
            dict: A dictionary of the form
                 {pit_track_id_1: {person_track_id_1: proximity_in pixels, ...}, ...}
        """
        time_ms = vignette.present_timestamp_ms
        if time_ms is None:
            logger.debug(
                "Vignette has no valid time to get proximity of PERSON and PIT"
            )
            return {}

        proximity_area_ratio = get_proximity_area_ratio(
            vignette, self.actor_categories
        )

        if proximity_area_ratio.distance_matrix is None:
            logger.debug(
                "Vignette has no valid PIT and PERSON tracks to get proximity"
            )
            return {}
        pit_proximity_dict = (
            PitPersonProximityAlgorithm._get_pit_proximity_dict(
                proximity_area_ratio.distance_matrix,
                proximity_area_ratio.track_ids_category_1,
                proximity_area_ratio.track_ids_category_2,
                vignette.tracklets,
                time_ms,
            )
        )
        return pit_proximity_dict
