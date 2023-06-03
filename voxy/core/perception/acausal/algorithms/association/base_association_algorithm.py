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

from typing import Dict, List

from core.structs.vignette import Vignette


class BaseAssociationAlgorithm:
    def __init__(self, config: dict, actor_categories: List) -> None:
        self._config = config
        self._actor_categories = actor_categories

    def _grab_required_tracklets(self, vignette: Vignette) -> dict:
        # Returns a mapping of required categories with tracklets in current vignette frame
        tracklet_mapping: Dict[int, list] = {}
        for category in self._actor_categories:
            tracklet_mapping[category] = []
        if bool(vignette.tracklets):
            for tracklet in vignette.tracklets.values():
                if tracklet.category in self._actor_categories:
                    tracklet_mapping[tracklet.category].append(
                        tracklet.track_id
                    )
        return tracklet_mapping

    def find_association(self, vignette: Vignette) -> dict:
        raise NotImplementedError(
            "Association algorithm must implement find_association."
        )
