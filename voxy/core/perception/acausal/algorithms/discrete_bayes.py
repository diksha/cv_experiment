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

import numpy as np
from filterpy.discrete_bayes import predict as discrete_bayes_predict
from filterpy.discrete_bayes import update as discrete_bayes_update

from core.perception.acausal.algorithms.base import BaseAcausalAlgorithm
from core.structs.actor import ActorCategory, DoorState


class DiscreteBayesAlgorithm(BaseAcausalAlgorithm):
    def __init__(self, config):
        self._config = config
        self._door_state_kernel = [0.05, 0.9, 0.05]

    def _update_door_state_probabilities(
        self, door_tracklet, present_timestamp_ms
    ):
        present_actor_index = door_tracklet.get_actor_index_at_time(
            present_timestamp_ms
        )

        # Return if present actor doesn't exists.
        if (
            present_actor_index is None
            or door_tracklet[present_actor_index] is None
        ):
            return

        # Return if actor right before present doesn't exists.
        if (
            present_actor_index - 1 < 0
            or door_tracklet[present_actor_index - 1] is None
        ):
            return

        door_tracklet[
            present_actor_index
        ].door_state_probabilities = discrete_bayes_predict(
            discrete_bayes_update(
                door_tracklet[present_actor_index].door_state_probabilities,
                np.asarray(
                    door_tracklet[
                        present_actor_index - 1
                    ].door_state_probabilities
                ),
            ),
            offset=0,
            kernel=self._door_state_kernel,
        ).tolist()

        max_index = np.argmax(
            door_tracklet[present_actor_index].door_state_probabilities
        )
        if max_index == 0:
            door_tracklet[
                present_actor_index
            ].door_state = DoorState.FULLY_OPEN
        elif max_index == 1:
            door_tracklet[
                present_actor_index
            ].door_state = DoorState.PARTIALLY_OPEN
        else:
            door_tracklet[
                present_actor_index
            ].door_state = DoorState.FULLY_CLOSED

    def process_vignette(self, vignette):
        for _, tracklet in vignette.tracklets.items():
            if tracklet.category is ActorCategory.DOOR:
                self._update_door_state_probabilities(
                    tracklet, vignette.present_timestamp_ms
                )
        return vignette
