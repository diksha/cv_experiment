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
from dataclasses import dataclass

import numpy as np

from core.structs.actor import ActorCategory
from core.structs.vignette import Vignette
from core.utils.struct_utils.utils import get_bbox_from_xysr


@dataclass
class ActorL2Distance:
    """
    Represents the distance between two actor classes in vignette.
    """

    distance_matrix: np.ndarray
    track_ids_category_1: np.ndarray
    track_ids_category_2: np.ndarray


@dataclass
class ActorArea:
    """
    Represents the area of an actor in vignette.
    """

    area_matrix: np.ndarray
    track_ids: np.ndarray


def get_centroid_l2_distance_matrix(
    actor_category_1: ActorCategory,
    actor_category_2: ActorCategory,
    vignette: Vignette,
) -> ActorL2Distance:
    """Computes L2 Matrix between centroids of a actor_category_1 type
    and all actors of actor_category_2 type.

    Args:
        actor_category_1: category of actors for which to compute L2 distance
        actor_category_2: category of actors for which to compute L2 distance
        vignette (Vignette): Vignette at present timestamp

    Returns:
        ActorL2Distance containing l2 distance between an actor of type actor_category_1
        and actor of type actor_category_2 and their corresponding track ids.
    """
    track_id_actor_1, xysr_actor_1 = vignette.filter_null_xysr_tracks(
        actor_category_1
    )
    track_id_actor_2, xysr_actor_2 = vignette.filter_null_xysr_tracks(
        actor_category_2
    )

    if len(xysr_actor_1) == 0 or len(xysr_actor_2) == 0:
        return ActorL2Distance(
            distance_matrix=None,
            track_ids_category_1=None,
            track_ids_category_2=None,
        )

    xysr_1 = np.array(xysr_actor_1)
    _x = xysr_1[:, 0:2]
    xysr_2 = np.array(xysr_actor_2)
    _y = xysr_2[:, 0:2]
    x_l2_norm = np.sum(np.square(_x), axis=1)[:, np.newaxis]
    y_l2_norm = np.sum(np.square(_y), axis=1)
    x_y_mul = np.dot(_x, _y.T)
    squared_distance = x_l2_norm + y_l2_norm - (2 * x_y_mul)

    return ActorL2Distance(
        distance_matrix=np.sqrt(squared_distance),
        track_ids_category_1=np.array(track_id_actor_1),
        track_ids_category_2=np.array(track_id_actor_2),
    )


def get_center_bottom_l2_distance_matrix(
    actor_category_1: ActorCategory,
    actor_category_2: ActorCategory,
    vignette: Vignette,
) -> ActorL2Distance:
    """Computes L2 Matrix between center bottom of a actor_category_1 type
    and all actors of actor_category_2 type.

    Args:
        actor_category_1: category of actors to compute L2 distance for
        actor_category_2: category of actors to compute L2 distance for
        vignette (Vignette): Vignette at present timestamp

    Returns:
        ActorL2Distance containing l2 distance between an actor of type actor_category_1
        and actor of type actor_category_2 and their corresponding track ids.
    """
    track_id_actor_1, xysr_actor_1 = vignette.filter_null_xysr_tracks(
        actor_category_1
    )
    track_id_actor_2, xysr_actor_2 = vignette.filter_null_xysr_tracks(
        actor_category_2
    )

    if len(xysr_actor_1) == 0 or len(xysr_actor_2) == 0:
        return ActorL2Distance(
            distance_matrix=None,
            track_ids_category_1=None,
            track_ids_category_2=None,
        )

    xysr_1 = np.array(xysr_actor_1)
    bbox_1 = get_bbox_from_xysr(xysr_1)
    _x = np.concatenate(
        (
            xysr_1[:, 0].reshape(xysr_1.shape[0], 1),
            bbox_1[:, 1].reshape(xysr_1.shape[0], 1),
        ),
        axis=1,
    )
    xysr_2 = np.array(xysr_actor_2)
    bbox_2 = get_bbox_from_xysr(xysr_2)
    _y = np.concatenate(
        (
            xysr_2[:, 0].reshape(xysr_2.shape[0], 1),
            bbox_2[:, 1].reshape(xysr_2.shape[0], 1),
        ),
        axis=1,
    )
    x_l2_norm = np.sum(np.square(_x), axis=1)[:, np.newaxis]
    y_l2_norm = np.sum(np.square(_y), axis=1)
    x_y_mul = np.dot(_x, _y.T)
    squared_distance = x_l2_norm + y_l2_norm - (2 * x_y_mul)

    return ActorL2Distance(
        distance_matrix=np.sqrt(squared_distance),
        track_ids_category_1=np.array(track_id_actor_1),
        track_ids_category_2=np.array(track_id_actor_2),
    )


def get_area_array(
    actor_category: ActorCategory, vignette: Vignette
) -> ActorArea:
    """Get all areas of particular category

    Args:
        actor_category (ActorCategory): Get the scale value for all nonnull tracks of category
        vignette (Vignette): Vignette at present timestamp

    Returns:
        tuple: track ids with valid tracks, actor scales (areas)
    """
    track_ids_actor, xysr_actor = vignette.filter_null_xysr_tracks(
        actor_category
    )
    if len(track_ids_actor) == 0:
        return ActorArea(area_matrix=None, track_ids=None)
    return ActorArea(
        area_matrix=np.array(xysr_actor)[:, 2],
        track_ids=np.array(track_ids_actor),
    )


def get_l2_distance_matrix(
    vignette: Vignette,
    actor_categories: typing.List[ActorCategory],
    distance_function: str = "centroid",
) -> ActorL2Distance:
    """Get the l2 distance matrix of actors in the two actor categories

    Args:
        vignette (Vignette): Vignette at present timestamp
        actor_categories (typing.List[ActorCategory]): Two actor categories
        distance_function (str, optional): Distance function to use. Defaults to "centroid".

    Returns:
        actor_distance(ActorL2Distance): information on actor distance and valid actors
    """
    if distance_function == "centroid":
        actor_distance = get_centroid_l2_distance_matrix(
            *actor_categories, vignette=vignette
        )
    elif distance_function == "center_bottom":
        actor_distance = get_center_bottom_l2_distance_matrix(
            *actor_categories, vignette=vignette
        )
    return actor_distance


def get_sum_area_matrix(
    vignette: Vignette,
    actor_categories: typing.List[ActorCategory],
) -> np.ndarray:
    """Get the sum of the areas of two actor categories as a matrix

    Args:
        vignette (Vignette): Vignette at timestamp of interest
        actor_categories (typing.List[ActorCategory]): Two actor categories

    Returns:
        np.ndarray: An array of summed area: rows (actor_category_1), columns (actor_category_2)
    """
    area_1 = get_area_array(actor_categories[0], vignette)
    area_2 = get_area_array(actor_categories[1], vignette)

    area_actor_1 = area_1.area_matrix
    area_actor_2 = area_2.area_matrix
    if area_actor_1 is None or area_actor_2 is None:
        return None

    area_actor_1 = area_actor_1[:, np.newaxis]
    area_actor_2 = np.broadcast_to(
        area_actor_2.T, (area_actor_1.shape[0], area_actor_2.shape[0])
    )

    area_matrix = area_actor_1 + area_actor_2
    return area_matrix


def get_proximity_area_ratio(
    vignette: Vignette,
    actor_categories: typing.List[ActorCategory],
) -> ActorL2Distance:
    """Get the distance matrix normalized by the area of the two objects

    Args:
        vignette (Vignette): Vignette at the present timestamp
        actor_categories (typing.List[ActorCategory]): Two actor categories

    Returns:
        tuple: (proximity to area ratio, valid actors in category 1, valid actors in category 2)
    """
    actor_distances = get_l2_distance_matrix(vignette, actor_categories)
    area_matrix = get_sum_area_matrix(vignette, actor_categories)
    if area_matrix is None or actor_distances.distance_matrix is None:
        return ActorL2Distance(None, None, None)
    proximity_area_ratio_matrix = actor_distances.distance_matrix / area_matrix
    return ActorL2Distance(
        distance_matrix=proximity_area_ratio_matrix,
        track_ids_category_1=actor_distances.track_ids_category_1,
        track_ids_category_2=actor_distances.track_ids_category_2,
    )
