#
# Copyright 2020-2022 Voxel Labs, Inc.
# All rights reserved.
#
# This document may not be reproduced, republished, distributed, transmitted,
# displayed, broadcast or otherwise exploited in any manner without the express
# prior written permission of Voxel Labs, Inc. The receipt or possession of this
# document does not convey any rights to reproduce, disclose, or distribute its
# contents, or to manufacture, use, or sell anything that it may describe, in
# whole or in part.
#

from core.perception.acausal.algorithms.association.pit_person_association import (
    PitPersonAssociationAlgorithm,
)
from core.perception.acausal.algorithms.base import BaseAcausalAlgorithm
from core.structs.actor import ActorCategory
from core.structs.tracklet import Tracklet
from core.structs.vignette import Vignette

# TODO (Gabriel): Smoothing params via config
K_LOOKBACK_TIME_MS = 5000
K_MIN_FRACTION_TO_ASSOCIATE = 0.5


class AssociationAlgorithmController(BaseAcausalAlgorithm):
    def __init__(self, config: dict) -> None:
        self._config = config
        self._association_algorithms = [
            PitPersonAssociationAlgorithm(config),
        ]

    def _update_person_to_pit_metadata(
        self, timestamp_ms: int, tracklet: Tracklet
    ) -> None:
        smoothed_association = tracklet.get_smoothed_association_for_actor(
            ActorCategory.PIT,
            timestamp_ms,
            K_LOOKBACK_TIME_MS,
            K_MIN_FRACTION_TO_ASSOCIATE,
        )
        if (
            smoothed_association is not None
            and not tracklet.is_associated_with_pit
        ):
            tracklet.is_associated_with_pit = True
        elif smoothed_association is None and tracklet.is_associated_with_pit:
            tracklet.is_associated_with_pit = False

    def _update_pit_to_person_metadata(
        self, timestamp_ms: int, tracklet: Tracklet
    ) -> None:
        smoothed_association = tracklet.get_smoothed_association_for_actor(
            ActorCategory.PERSON,
            timestamp_ms,
            K_LOOKBACK_TIME_MS,
            K_MIN_FRACTION_TO_ASSOCIATE,
        )
        if (
            smoothed_association is not None
            and not tracklet.is_associated_with_person
        ):
            tracklet.is_associated_with_person = True
        elif (
            smoothed_association is None and tracklet.is_associated_with_person
        ):
            tracklet.is_associated_with_person = False

    def _compute_association_metadata_for_tracklet(
        self, timestamp_ms: int, tracklet: Tracklet
    ) -> None:
        if tracklet.category == ActorCategory.PERSON:
            self._update_person_to_pit_metadata(timestamp_ms, tracklet)
        elif tracklet.category == ActorCategory.PIT:
            self._update_pit_to_person_metadata(timestamp_ms, tracklet)

    def process_vignette(self, vignette: Vignette) -> Vignette:
        frame_timestamp_ms = vignette.present_timestamp_ms
        for algorithm in self._association_algorithms:
            associations_dict: dict = algorithm.find_association(vignette)
            if associations_dict:
                for (
                    association_key,
                    association_matrix,
                ) in associations_dict.items():
                    for track_id, asssociated_id in association_matrix:
                        tracklet = vignette.tracklets[track_id]
                        tracklet.update_tracklet_associations(
                            association_key,
                            frame_timestamp_ms,
                            asssociated_id,
                        )
                        self._compute_association_metadata_for_tracklet(
                            frame_timestamp_ms, tracklet
                        )
        return vignette
