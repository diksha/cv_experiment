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

import numpy as np
from loguru import logger
from scipy.optimize import linear_sum_assignment

from core.perception.acausal.algorithms.association.base_association_algorithm import (
    BaseAssociationAlgorithm,
)
from core.structs.actor import ActorCategory
from core.structs.vignette import Vignette
from core.utils.struct_utils.utils import get_bbox_from_xysr

# TODO (Gabriel): Association threshold tuning
K_MIN_IOP_ASSOCIATION = 0.5
K_MAX_L2_DISTANCE_ASSOCIATION = 500
K_MAX_ASSOCIATION_SCORE = K_MAX_L2_DISTANCE_ASSOCIATION / K_MIN_IOP_ASSOCIATION
K_EPSILON = 0.0001


class PitPersonAssociationAlgorithm(BaseAssociationAlgorithm):
    def __init__(self, config: dict) -> None:
        super().__init__(config, [ActorCategory.PERSON, ActorCategory.PIT])

    def _get_iop_matrix(
        self, person_xysrs: np.array, pit_xysrs: np.ndarray
    ) -> np.ndarray:
        """
        Vecotorized get_iop for person and pit arrays

        Returns:
            np.ndarray of iop between pits and people
        """
        person_x1, person_y1, person_x2, person_y2 = np.split(
            get_bbox_from_xysr(person_xysrs), 4, 1
        )
        pit_x1, pit_y1, pit_x2, pit_y2 = np.split(
            get_bbox_from_xysr(pit_xysrs), 4, 1
        )
        person_area = (person_x2 - person_x1) * (person_y2 - person_y1)
        x1 = np.maximum(person_x1, pit_x1.T)
        y1 = np.maximum(person_y1, pit_y1.T)
        x2 = np.minimum(person_x2, pit_x2.T)
        y2 = np.minimum(person_y2, pit_y2.T)
        intersection_area = (x2 - x1) * (y2 - y1)
        valid_iop = np.logical_and(x1 <= x2, y1 <= y2)
        iop_matrix = np.where(
            valid_iop, intersection_area / person_area, K_EPSILON
        )
        return iop_matrix

    def _get_l2_distance_matrix(self, X: np.array, Y: np.array) -> np.ndarray:
        """
        Vectorized l2 distance between two actor arrays

        Returns:
            np.ndarray of l2 distances between actors
        """
        x_l2_norm = np.sum(np.square(X), axis=1)[:, np.newaxis]
        y_l2_norm = np.sum(np.square(Y), axis=1)
        x_y_mul = np.dot(X, Y.T)
        squared_distance = x_l2_norm + y_l2_norm - (2 * x_y_mul)
        return np.sqrt(squared_distance)

    def _generate_associations_for_missing_actor_category(
        self,
        person_track_ids: list,
        pit_track_ids: list,
    ) -> dict:
        """
        This function is used whenever we have tracklets for
        either persons or pits. Since one actor category is
        missing, the associations for the nonnull tracklets
        are None

        Args:
            list of person track ids and pit track ids

        Returns:
            dictionary of associations
        """
        association_map = {}
        count_persons = len(person_track_ids)
        count_pits = len(pit_track_ids)
        n_tracks = np.max([count_persons, count_pits])
        associations = np.empty([n_tracks, 2], dtype=object)
        # If count_persons > 0, count_pits == 0, so give associations
        # to only persons, else do the opposite
        if count_persons > 0:
            associations[:, 0] = person_track_ids
            associations[:, 1] = None
            association_map[ActorCategory.PIT] = associations
        else:
            associations[:, 0] = pit_track_ids
            associations[:, 1] = None
            association_map[ActorCategory.PERSON] = associations
        return association_map

    def _get_valid_xysr_tracks(
        self, vignette: Vignette, track_id_list: list
    ) -> tuple:
        """
        Args:
            vignette and list of relevant track ids for
            one actor

        Returns:
            list of track ids that have nonnull xysr
            list of track ids with null xysr
            list of nonnull xysrs
        """
        time_ms = vignette.present_timestamp_ms
        valid_tracklets = []
        nonnull_xysr = []
        # For each specific tracklet corresponding to a single
        # actor, if xysr is missing (ie smoothing issue), remove
        # from valid actor list (since we cannot compute metrics)
        for track_id in track_id_list:
            tracklet = vignette.tracklets[track_id]
            xysr = tracklet.get_xysr_at_time(time_ms)
            if xysr is not None:
                valid_tracklets.append(track_id)
                nonnull_xysr.append(xysr)
        return np.array(valid_tracklets), np.array(nonnull_xysr)

    def _generate_associations_for_all_invalid_actors(
        self,
        person_track_ids: list,
        pit_track_ids: list,
    ) -> dict:
        """
        This function is used when we have both actors
        present in the frame, but one actor category has
        no valid xysr and thus all tracklets have None
        associations

        Args:
            list of person track ids and pit track ids

        Returns:
            dictionary of associaitions
        """
        association_map = {}
        count_persons = len(person_track_ids)
        count_pits = len(pit_track_ids)
        person_associations = np.empty([count_persons, 2], dtype=object)
        person_associations[:, 0] = person_track_ids
        person_associations[:, 1] = None
        association_map[ActorCategory.PIT] = person_associations
        pit_associations = np.empty([count_pits, 2], dtype=object)
        pit_associations[:, 0] = pit_track_ids
        pit_associations[:, 1] = None
        association_map[ActorCategory.PERSON] = pit_associations
        return association_map

    def _compute_association_score_matrix(
        self, person_xysr: np.array, pit_xysr: np.array
    ) -> np.ndarray:
        """
        This function computes the score matrix of l2/iop

        Args:
            list of person xysrs and pit xysrs

        Returns:
            np.ndarray of scores for each person to each pit
        """
        person_center_points = person_xysr[:, 0:2]
        pit_center_points = pit_xysr[:, 0:2]
        iop_matrix = self._get_iop_matrix(person_xysr, pit_xysr)
        l2_matrix = self._get_l2_distance_matrix(
            person_center_points, pit_center_points
        )
        l2_iop_ratio = np.divide(l2_matrix, iop_matrix)
        return l2_iop_ratio

    def _get_associated_rows_and_columns(
        self,
        l2_iop_ratio: np.ndarray,
    ) -> tuple:
        """
        This function computes the rows and columns of the l2
        iop ratio array where we have at least one valid association
        score, to avoid scenarios where the hungarian algorithm
        assumes all rows and columns must be associated

        Args:
            l2 iop ratio matrix

        Returns:
            np.array of rows and columns where at least one score
            is below our max association score
        """
        associated_rows = []
        for row in range(l2_iop_ratio.shape[0]):
            scores = l2_iop_ratio[row, :]
            if np.min(scores) < K_MAX_ASSOCIATION_SCORE:
                associated_rows.append(row)

        associated_columns = []
        for col in range(l2_iop_ratio.shape[1]):
            scores = l2_iop_ratio[:, col]
            if np.min(scores) < K_MAX_ASSOCIATION_SCORE:
                associated_columns.append(col)

        return np.array(associated_rows), np.array(associated_columns)

    def _generate_association_array(
        self,
        n_total_actors: int,
        valid_track_ids: np.array,
        invalid_track_ids: np.array,
        valid_associated_ids: np.array,
        best_track_idx: np.array,
        best_associated_idx: np.array,
    ) -> np.ndarray:
        """
        This function will generate the association array where
        the rows correspond to each tracklet in the frame for a specific
        actor type, column 0 corresponds to the track id of the actor in
        question, and column 1 corresponds to the track id associated to
        the actor in column 0

        Args:
            int of the total number of actors to associate, list of the
            valid actors to associate, list of the invalid actors which
            will not have associations, np.array idx of best scoring actors,
            np.array corresponding idx of best scoring associated actor, and
            np.array of best scores

        Returns:
            np.ndarray of size n_total_actors, 2 where column 0 are track ids for
            actors we want to generate associations for, and column 1 are track
            ids for the actors that have the best association score for
        """
        association_array = np.empty([n_total_actors, 2], dtype=object)
        for i, track_id in enumerate(valid_track_ids):
            assigned_track_id = None
            if i in best_track_idx:
                loc = np.where(best_track_idx == i)
                assigned_associated_idx = best_associated_idx[loc][0]
                assigned_track_id = valid_associated_ids[
                    assigned_associated_idx
                ]
            association_array[i, :] = [
                track_id,
                assigned_track_id,
            ]

        if invalid_track_ids.size > 0:
            null_actor = np.isnan(association_array[:, 0].astype(float))
            association_array[null_actor, 0] = invalid_track_ids
        return association_array

    def find_association(self, vignette: Vignette) -> dict:
        """
        Finds the associations for pits and people for the current
        frame

        Args:
            vignette: current vignette

        Returns:
            dictionary where the key is the actor category corresponding
            to the actors the associated actors (not the actor category
            we want to generate associations for) and the values is an
            np.array of size n x 2, where n is the number of actors we
            want to generate associations for.

        Example:
            {ActorCategory.PIT: [101, 102]}
            This means that actor 101 is associated to actor 102, and
            actor 102 is a PIT
        """
        time_ms = vignette.present_timestamp_ms
        if time_ms is None:
            logger.debug(
                "Vignette has no valid time to associate PERSON and PIT"
            )
            return {}

        tracklet_id_map = super()._grab_required_tracklets(vignette)
        if not tracklet_id_map:
            logger.debug("Vignette has no tracklets")
            return {}

        all_person_tracklet_ids = np.array(
            tracklet_id_map[ActorCategory.PERSON]
        )
        all_pit_tracklet_ids = np.array(tracklet_id_map[ActorCategory.PIT])
        if (
            not all_person_tracklet_ids.size > 0
            or not all_pit_tracklet_ids.size > 0
        ):
            logger.debug(
                f"Vignette missing required actors, \
                 PERSON, {all_person_tracklet_ids.size > 0}, \
                 PIT, {all_pit_tracklet_ids.size > 0}"
            )
            return self._generate_associations_for_missing_actor_category(
                all_person_tracklet_ids, all_pit_tracklet_ids
            )

        (
            valid_person_tracklet_ids,
            nonnull_person_xysr,
        ) = self._get_valid_xysr_tracks(vignette, all_person_tracklet_ids)
        (
            valid_pit_tracklet_ids,
            nonnull_pit_xysr,
        ) = self._get_valid_xysr_tracks(vignette, all_pit_tracklet_ids)

        if not nonnull_person_xysr.size > 0 or not nonnull_pit_xysr.size > 0:
            logger.debug(
                f"Could not interpolate tracklet xysr, \
                PERSON, {nonnull_person_xysr.size > 0}, \
                PIT, {nonnull_pit_xysr.size > 0}"
            )
            return self._generate_associations_for_all_invalid_actors(
                all_person_tracklet_ids, all_pit_tracklet_ids
            )

        l2_iop_ratio = self._compute_association_score_matrix(
            nonnull_person_xysr, nonnull_pit_xysr
        )
        valid_rows, valid_columns = self._get_associated_rows_and_columns(
            l2_iop_ratio
        )
        if not valid_rows.size > 0 or not valid_columns.size > 0:
            logger.debug(
                f"No valid association scores, \
                PERSON, {valid_rows.size > 0}, \
                PIT, {valid_columns.size > 0}"
            )
            return self._generate_associations_for_all_invalid_actors(
                all_person_tracklet_ids, all_pit_tracklet_ids
            )

        # Update index of valid track ids to associate
        l2_iop_ratio = l2_iop_ratio[valid_rows, :][:, valid_columns]
        valid_person_tracklet_ids = valid_person_tracklet_ids[valid_rows]
        invalid_person_tracklet_ids = all_person_tracklet_ids[
            ~np.isin(all_person_tracklet_ids, valid_person_tracklet_ids)
        ]
        valid_pit_tracklet_ids = valid_pit_tracklet_ids[valid_columns]
        invalid_pit_tracklet_ids = all_pit_tracklet_ids[
            ~np.isin(all_pit_tracklet_ids, valid_pit_tracklet_ids)
        ]

        best_person_idx, best_pit_idx = linear_sum_assignment(l2_iop_ratio)

        association_map = {}
        count_persons = len(all_person_tracklet_ids)
        count_pits = len(all_pit_tracklet_ids)
        person_associations = self._generate_association_array(
            count_persons,
            valid_person_tracklet_ids,
            invalid_person_tracklet_ids,
            valid_pit_tracklet_ids,
            best_person_idx,
            best_pit_idx,
        )
        association_map[ActorCategory.PIT] = person_associations
        pit_associations = self._generate_association_array(
            count_pits,
            valid_pit_tracklet_ids,
            invalid_pit_tracklet_ids,
            valid_person_tracklet_ids,
            best_pit_idx,
            best_person_idx,
        )
        association_map[ActorCategory.PERSON] = pit_associations

        return association_map
