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

import typing
from copy import deepcopy

import cv2
import numpy as np

from core.common.functional.lib.transform import Transform
from core.common.functional.registry.registry import TransformRegistry
from core.datasets.generators.association import PersonPpeAssociation
from core.incidents.utils import CameraConfig
from core.structs.actor import (
    Actor,
    ActorCategory,
    DoorState,
    get_ordered_actors,
    get_track_uuid,
)
from core.structs.attributes import RectangleXCYCWH, RectangleXYWH
from core.structs.frame import Frame

# trunk-ignore-all(pylint/W0221)


@TransformRegistry.register()
class GaussianBlur(Transform):
    """
    Applies gaussian blur to an image when called.
    """

    def __init__(
        self,
        ksize: tuple,
        sigma_x: int,
        sigma_y: int = 0,
        border_type: int = cv2.BORDER_DEFAULT,
    ):
        self.ksize = ksize
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.border_type = border_type

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Executes transform to apply gaussian blur to an image

        Args:
            image (np.ndarray): The image to apply gaussian blur

        Returns:
            np.ndarray: blurred image
        """
        return cv2.GaussianBlur(
            image,
            self.ksize,
            self.sigma_x,
            self.sigma_y,
            self.border_type,
        )


@TransformRegistry.register()
class MotionBlur(Transform):
    """
    Applies motion blur to an image when called
    """

    def __init__(self, blur_intensity: int = 0.01):
        self.blur_intensity = blur_intensity

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Exeutes transform to apply motion blur to an image

        Args:
            image (np.ndarray): The image to apply motion blur

        Returns:
            np.ndarray: blurred image
        """
        height, width, _ = image.shape

        kernel_len_h = int(height * self.blur_intensity)
        kernel_len_w = int(width * self.blur_intensity)

        kernel_h = np.ones((kernel_len_h, 1), dtype=np.float32) / kernel_len_h
        kernel_w = np.ones((1, kernel_len_w), dtype=np.float32) / kernel_len_w

        if np.random.random_sample() > 0.5:
            return cv2.filter2D(image, -1, kernel_h)
        return cv2.filter2D(image, -1, kernel_w)


@TransformRegistry.register()
class GetAttribute(Transform):
    """
    Gets the attribute from the object
    """

    def __init__(self, attribute):
        self.attribute = attribute

    def __call__(self, item: typing.Any) -> typing.Any:
        """Calling function for getting attributes

        Args:
            item (typing.Any): object to get attribute from

        Returns:
            typing.Any: attribute details
        """

        return getattr(item, self.attribute)


@TransformRegistry.register()
class GetObservation(Transform):
    """
    Label and data transforms both are given a `data` object and
    a `observation` object. This returns the observation object.
    See `__call__` for more details
    """

    def __init__(self):
        pass

    def __call__(
        self, data: typing.Any, observation: typing.Any
    ) -> typing.Any:
        """Calling function to get observation object

        Args:
            data (typing.Any): data object for labels
            observation (typing.Any): observation object for transforms

        Returns:
            typing.Any: observation object for labels and transforms
        """
        return observation


@TransformRegistry.register()
class GetData(Transform):
    """
    Label and data transforms both are given a `data` object and
    a `observation` object. This returns the data object.
    See `__call__` for more details
    """

    def __init__(self):
        pass

    def __call__(
        self, data: typing.Any, observation: typing.Any
    ) -> typing.Any:
        """Calling function to get data

        Args:
            data (typing.Any): data with labels
            observation (typing.Any): observation details

        Returns:
            typing.Any: gets the data given data and observation
        """
        return data


@TransformRegistry.register()
class CropFromActor(Transform):
    """
    Crop the actor given its polygon from the image
    """

    def __init__(self):
        pass

    def __call__(self, image: np.array, actor: Actor) -> np.array:
        """
        Crops the image based on the polygon of the actor

        Args:
            image (np.array): the input image to be cropped
            actor (Actor): the actor to get the polygon from

        Returns:
            np.array: the cropped image
        """
        rectangle = RectangleXYWH.from_polygon(actor.polygon)
        cropped_image = image[
            rectangle.top_left_vertice.y : rectangle.top_left_vertice.y
            + rectangle.h,
            rectangle.top_left_vertice.x : rectangle.top_left_vertice.x
            + rectangle.w,
        ]
        return cropped_image


@TransformRegistry.register()
class Resize(Transform):
    """
    Resize the image to the new size using cv2.resize
    """

    def __init__(self, size: tuple):
        self.size = size

    def __call__(self, image: np.array) -> np.array:
        """Resize image

        Args:
            image (np.array): Image to resize

        Returns:
            np.array: resized image
        """

        return cv2.resize(image, self.size, interpolation=cv2.INTER_AREA)


@TransformRegistry.register()
class FilterActorsByCategory(Transform):
    """
    Filter all actors not in the categories given
    """

    def __init__(self, categories: typing.List):
        self.categories = [ActorCategory[category] for category in categories]

    def __call__(self, actors: typing.List[Actor]) -> typing.List[Actor]:
        """Filters actors by category

        Args:
            actors (typing.List[Actor]): list of actors to filter

        Returns:
            typing.List[Actor]: filtered actors
        """
        return filter(lambda actor: actor.category in self.categories, actors)


@TransformRegistry.register()
class ExcludeActorsByDoorState(Transform):
    """
    Excludes any actors with the state in `states` parameter
    """

    def __init__(self, states: typing.List) -> None:
        """
        Filters any actor that have the specified door state

        Args:
            states (typing.List): the list of states to exclude
        """
        self.excluded_states = [DoorState[state] for state in states]

    def __call__(self, actors: typing.List[Actor]) -> typing.List[Actor]:
        """
        Returns an iterable filter given a list of actors

        Args:
            actors (typing.List[Actor]): the actors to filter

        Returns:
            typing.List[Actor]: the filtered list of actors
        """
        return filter(
            lambda actor: actor.door_state not in self.excluded_states, actors
        )


@TransformRegistry.register()
class FilterActorsByUUID(Transform):
    """
    Filters the actors based on their uuid
    """

    def __init__(self, uuids):
        self.uuids = uuids

    def __call__(self, actors: typing.List[Actor]) -> typing.List[Actor]:
        """Given uuid filters the actors

        Args:
            actors (typing.List[Actor]): Actors to filter on

        Returns:
            typing.List[Actor]: Filtered actors
        """
        return filter(lambda actor: actor.track_uuid in self.uuids, actors)


@TransformRegistry.register()
class FilterDoorActorsFromCameraConfig(Transform):
    """
    Filters door actors using only valid ones from the camera config. This
    generates a list of valid door track uuids and filters all actors that are
    not contained in the camera config
    """

    def __init__(self, camera_uuid):
        camera_config = CameraConfig(camera_uuid, -1, -1)
        door_uuids = [
            get_track_uuid(camera_uuid, str(door.door_id), ActorCategory.DOOR)
            for door in camera_config.doors
        ]
        self.uuids = door_uuids

    def __call__(self, actors: typing.List[Actor]) -> typing.List[Actor]:
        """
        Executes transform to filter based on uuids from camera config

        Args:
            actors (typing.List[Actor]): the actors to be filtered

        Returns:
            typing.List[Actor]: the filtered map of actors
        """
        return filter(lambda actor: actor.track_uuid in self.uuids, actors)


@TransformRegistry.register()
class ConvertBoolToName(Transform):
    """
    Converts the input boolean into a given `true_name`
    or a `false_name`. This is useful when converting
    between boolean labels to human readable names

    If the item is None, then `none_name` is used
    """

    def __init__(
        self, true_name: str, false_name: str, none_name: str = "UNKNOWN"
    ):
        """
        Initializes the conversion transform

        Args:
            true_name (str): the name to be returned if the item is True
            false_name (str): the name to be returned if the item is False
            none_name (str, optional): the name to be returned if the item is None.
                                       Defaults to "UNKNOWN".
        """
        self.true_name = true_name
        self.false_name = false_name
        self.none_name = none_name

    def __call__(self, item: bool) -> str:
        """
        Converts the input boolean into a given `true_name`
        or a `false_name`. This is useful when converting
        between boolean labels to human readable names

        If the item is None, then `none_name` is used

        Args:
            item (bool): the boolean item to convert

        Returns:
            str: the output name
        """
        if item is None:
            return self.none_name
        return self.true_name if item else self.false_name


@TransformRegistry.register()
class FilterActorsWithNoneAttribute(Transform):
    """
    Filters an object based on if an attribute is None
    """

    def __init__(self, attribute_name: str):
        self.attribute_name = attribute_name

    def __call__(self, actors: typing.List[Actor]) -> typing.List[Actor]:
        """
        Returns a list of actors that do not have None for the given attribute name

        For example, if you were looking to filter any pits that do not have "forks raised"
        labeled (None) then you would just pass in: attribute_name = "forks_raised" and
        the none attributes actors would be filtered

        Args:
            actors (typing.List[Actor]): the actors to be filtered

        Returns:
            typing.List[Actor]: the filtered map of actors
        """
        return [
            actor
            for actor in actors
            if not getattr(actor, self.attribute_name) is None
        ]


@TransformRegistry.register()
class AssociatePPEFromFrame(Transform):
    """
    Performs association on the bounding boxes of PPE and the person
    and modifies the frame so the actors have the given attribute (bool)
    """

    def __init__(
        self, person_class, ppe_class, no_ppe_class, in_place, attribute
    ):
        self.associator = PersonPpeAssociation(
            iou_threshold=0,
            person_class=ActorCategory[person_class],
            ppe_actor_class=ActorCategory[ppe_class],
            no_ppe_actor_class=ActorCategory[no_ppe_class],
        )
        self.mutate = in_place
        self.attribute = attribute

    def __call__(self, frame: Frame) -> Frame:
        """Associates PPE from frame

        Args:
            frame (Frame): Frame to associate in

        Returns:
            Frame: PPE associated frame
        """
        association = self.associator.get_ppe_person_association(frame)[0]
        if association is None:
            return frame

        if self.mutate:
            return_frame = frame
        else:
            return_frame = deepcopy(frame)

        for actor in return_frame.actors:
            setattr(
                actor,
                self.attribute,
                association.get(actor.track_id),
            )
        return return_frame


@TransformRegistry.register()
class GetLabelYOLO(Transform):
    """
    Gets YOLO style labels [class, norm_xc, norm_yc, norm_w, norm_h]
    """

    def __init__(self, actor_categories: typing.List[str]):
        self.yolo_detector_classes = get_ordered_actors(actor_categories)

    def __call__(self, image: np.array, actor: Actor) -> str:
        """Get YOLO style labels

        Args:
            image (np.array): Image array actor exists in
            actor (Actor): Actor to generate labels for

        Returns:
            str: yolo labels
        """
        frame_height, frame_width, _ = image.shape
        (x_center, y_center, width, height) = RectangleXCYCWH.from_polygon(
            actor.polygon
        ).to_list()
        yolo_class = (
            self.yolo_detector_classes.index(actor.category)
            if actor.category in self.yolo_detector_classes
            else -1
        )
        norm_xc = float(x_center) / frame_width
        norm_yc = float(y_center) / frame_height
        norm_w = float(width) / frame_width
        norm_h = float(height) / frame_height
        return f"{yolo_class} {norm_xc} {norm_yc} {norm_w} {norm_h}"
