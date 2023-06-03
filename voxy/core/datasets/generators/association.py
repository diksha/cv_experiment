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
from typing import Tuple

import numpy as np
from loguru import logger

from core.structs.actor import ActorCategory
from core.structs.frame import Frame


class PersonPpeAssociation:
    def __init__(
        self,
        iou_threshold=0.2,
        person_class=ActorCategory.PERSON_V2,
        ppe_actor_class=ActorCategory.SAFETY_VEST,
        no_ppe_actor_class=ActorCategory.BARE_CHEST,
    ):
        logger.warning(
            (
                "This class will be deprecated when the new "
                "PPE labeling standard is finalized and "
                "implemented"
            )
        )
        self._iou_threshold = iou_threshold
        self.ppe_actor_class = ppe_actor_class
        self.no_ppe_actor_class = no_ppe_actor_class
        self.person_class = person_class

    def _iou(self, ppe_poly, person_poly):
        # Not true IOU. This represents how much of the
        # area of PPE label is contained within PERSON label.
        return ppe_poly.intersection(person_poly).area / ppe_poly.area

    def _get_association_score(
        self, person_actors: list, ppe_actors: list
    ) -> np.ndarray:
        if not ppe_actors:
            return np.zeros((len(person_actors), 1))
        person_polys = np.array(
            [actor.get_shapely_polygon() for actor in person_actors]
        )
        ppe_polys = np.array(
            [actor.get_shapely_polygon() for actor in ppe_actors]
        )

        iou_vectorized = np.vectorize(self._iou)

        iou_matrix = iou_vectorized(ppe_polys[None, :], person_polys[:, None])
        return iou_matrix

    def _get_pairs(
        self,
        ppe_iou_matrix: np.ndarray,
        no_ppe_iou_matrix: np.ndarray,
        frame_struct: Frame,
        person_ids: dict,
        ppe_ids: dict,
        no_ppe_ids: dict,
    ) -> Tuple[dict, dict]:
        person_ppe = {}
        person_ppe_id = {}
        for person_id, ppe_associations in enumerate(ppe_iou_matrix):
            has_ppe = None
            no_ppe_associations = no_ppe_iou_matrix[person_id]

            ppe_id = np.argmax(ppe_associations)
            no_ppe_id = np.argmax(no_ppe_associations)

            if len(no_ppe_associations > self._iou_threshold) == 0:
                has_ppe = True

            elif len(ppe_associations > self._iou_threshold) == 0:
                has_ppe = False

            else:
                # This is a tricky case. This means that a single person seems to have
                # positive associations with both no_ppe and ppe labels.
                # Let's go by max iou for now. This may fail in extreme edge cases.
                has_ppe = np.max(ppe_iou_matrix[person_id]) > np.max(
                    no_ppe_iou_matrix[person_id]
                )

            person_ppe_id[
                frame_struct.actors[person_ids[person_id]].track_id
            ] = (
                ppe_ids[ppe_id] if has_ppe else no_ppe_ids.get(no_ppe_id, None)
            )

            person_ppe[
                frame_struct.actors[person_ids[person_id]].track_id
            ] = has_ppe
        return person_ppe, person_ppe_id

    def _get_actors(
        self, frame_struct: Frame, actor_category: ActorCategory
    ) -> Tuple[dict, list]:

        actor_ids = {}
        _actors = []
        actor_idx = 0

        for idx, actor in enumerate(frame_struct.actors):
            if actor.category == actor_category:
                _actors.append(actor)
                actor_ids[actor_idx] = idx
                actor_idx += 1
        return actor_ids, _actors

    def get_ppe_person_association(
        self, frame_struct: Frame
    ) -> Tuple[dict, dict]:

        person_ids, person_actors = self._get_actors(
            frame_struct, self.person_class
        )
        ppe_ids, ppe_actors = self._get_actors(
            frame_struct, self.ppe_actor_class
        )
        no_ppe_ids, no_ppe_actors = self._get_actors(
            frame_struct, self.no_ppe_actor_class
        )

        if not person_actors:
            return None, None

        if not ppe_actors and not no_ppe_actors:
            return {}, {}

        ppe_iou_matrix = self._get_association_score(person_actors, ppe_actors)
        no_ppe_iou_matrix = self._get_association_score(
            person_actors, no_ppe_actors
        )

        person_ppe, person_ppe_ids = self._get_pairs(
            ppe_iou_matrix,
            no_ppe_iou_matrix,
            frame_struct,
            person_ids,
            ppe_ids,
            no_ppe_ids,
        )

        return person_ppe, person_ppe_ids
