import copy
import typing

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Compose

from core.incidents.utils import CameraConfig, DoorCameraConfig
from core.structs.actor import (
    Actor,
    ActorCategory,
    DoorOrientation,
    DoorState,
    get_track_uuid,
)
from core.structs.attributes import Point, Polygon, RectangleXYWH


# pre-processing functions
def crop_and_recolor_door(
    frame: np.ndarray, door: DoorCameraConfig, bgr2rgb: bool
) -> np.ndarray:
    """Crop and recolor doors for inference.
    Args:
        frame (np.ndarray): numpy array for frame to crop
        door (DoorCameraConfig): door to crop
        bgr2rgb (bool): transform bgr to rgb
    Returns:
        np.ndarray: cropped image
    """
    rect = RectangleXYWH.from_polygon(door.polygon)
    cropped_image = frame[
        max(0, rect.top_left_vertice.y) : min(
            frame.shape[0], rect.top_left_vertice.y + rect.h
        ),
        max(0, rect.top_left_vertice.x) : min(
            frame.shape[1], rect.top_left_vertice.x + rect.w
        ),
    ]

    if bgr2rgb:
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)

    return cropped_image


def pad_polygon(door: DoorCameraConfig, padding: int) -> Polygon:
    """Pad door polygon for inference.
    Args:
        door (DoorCameraConfig): camera config for door
        padding (int): amount to pad door by
    Returns:
        Polygon: door polygon
    """
    door_polygon = copy.deepcopy(door.polygon)

    return Polygon(
        vertices=[
            Point(
                max(
                    door_polygon.get_top_left().x - padding,
                    0,
                ),
                max(
                    door_polygon.get_top_left().y - padding,
                    0,
                ),
            ),
            Point(
                door_polygon.get_top_right().x + padding,
                max(
                    door_polygon.get_top_right().y - padding,
                    0,
                ),
            ),
            Point(
                door_polygon.get_bottom_right().x + padding,
                door_polygon.get_bottom_right().y + padding,
            ),
            Point(
                max(
                    door_polygon.get_bottom_left().x - padding,
                    0,
                ),
                door_polygon.get_bottom_left().y + padding,
            ),
        ]
    )


def extract_preprocessed_door_image(
    frame: np.ndarray,
    door: DoorCameraConfig,
    preprocessing_transforms: dict,
    transforms: Compose,
    device: torch.device,
) -> torch.Tensor:
    """Preprocesses frame for door classifier inference.
    Args:
        frame (np.ndarray): input frame to process
        door (DoorCameraConfig): door info for inference
        preprocessing_transforms (dict): additional transform not specified in training
        transforms (Compose): composition of functional transforms from training
        device (torch.device): device to store input tensor for inference

    Returns:
        torch.Tensor: frame preprocessed for inference on a door as a tensor
    """
    bgr2rgb = preprocessing_transforms["bgr2rgb"]
    cropped_image = crop_and_recolor_door(frame, door, bgr2rgb)
    normalized_image = (
        transforms(Image.fromarray(cropped_image)).to(device).unsqueeze(0)
    )
    return normalized_image


def extact_door_configs(
    camera_uuid: str, frame_height: int, frame_width: int
) -> typing.List[DoorCameraConfig]:
    """Extract door info from camera_config.
    Args:
        camera_uuid (str): camera uuid of camera being processed
        frame_height (int): height of input frames
        frame_width (int): width of input frames
    Returns:
        typing.List[DoorCameraConfig]: list of DoorCameraConfig
    """
    camera_config = CameraConfig(camera_uuid, frame_height, frame_width)
    return camera_config.doors


def post_process_prediction(
    prediction: torch.Tensor,
    state2class: dict,
    track_id: int,
    door: DoorCameraConfig,
    postprocessing_transforms: dict,
    camera_uuid: str,
) -> Actor:
    """Post process door classifier predictions for use in perception system.
    Args:
        prediction (torch.Tensor): raw door classifier output
        state2class (dict): dictionary to convert between state and door class
        track_id (int): track id of door via order in camera config
        door (DoorCameraConfig): door information from camera config
        postprocessing_transforms (dict): postprocessing transforms for prediction
        camera_uuid (str): uuid of the camera
    Returns:
        Actor: voxel door actor
    """
    prediction = torch.softmax(prediction, dim=0).cpu()
    (
        closed_probability,
        open_probability,
        partially_open_probability,
    ) = list(prediction)
    index = int(np.ndarray.argmax(prediction.detach().numpy()))

    open_door = index == state2class["open"]
    partially_open_door = index == state2class["partially_open"]

    door_state = DoorState.FULLY_CLOSED
    if open_door:
        door_state = DoorState.FULLY_OPEN
    elif partially_open_door:
        door_state = DoorState.PARTIALLY_OPEN

    door_orientation = (
        DoorOrientation.SIDE_DOOR
        if door.orientation == "SIDE_DOOR"
        else DoorOrientation.FRONT_DOOR
    )

    # There are some faulty door crossing events generated
    # due to the detected PIT being slightly outside the
    # door label. This results in multiple entering/exiting
    # events. Padding is a quick fix that should reduce our
    # piggyback false positives. In the future, we should
    # change the bounding box points for the camera, but
    # this would also affect the performance of the classifier.
    padding = postprocessing_transforms["padding"]
    door_polygon = pad_polygon(door, padding)

    return Actor(
        category=ActorCategory.DOOR,
        track_id=track_id,
        track_uuid=get_track_uuid(
            camera_uuid=camera_uuid,
            unique_identifier=str(door.door_id),
            category=ActorCategory.DOOR,
        ),
        polygon=door_polygon,
        manual=False,
        confidence=float(open_probability) + float(partially_open_probability),
        door_state=door_state,
        door_orientation=door_orientation,
        door_state_probabilities=[
            float(open_probability),
            float(partially_open_probability),
            float(closed_probability),
        ],
    )
