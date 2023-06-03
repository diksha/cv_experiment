import typing

import numpy as np
import torch

from core.perception.detector_tracker.utils import letterbox
from third_party.byte_track.utils import postprocess


def preprocess_image(
    image_batch: torch.tensor, input_shape: tuple, device: torch.device
) -> torch.Tensor:
    """preprocess_image.

    preprocesses the image so it is ready for inference

    Args:
        image_batch (torch.tensor): raw input image
        input_shape (np.array): input shape to transorm image to
        device (torch.device): device to store input tensor for inference
    Returns:
        torch.Tensor: the preprocessed image tensor
    """
    letterboxed_batch, scale, offset = map(
        list,
        zip(
            *[
                letterbox(image, new_shape=input_shape, auto=False)
                for _, image in enumerate(image_batch.numpy())
            ]
        ),
    )
    rgb_nchw = np.ascontiguousarray(
        np.array(letterboxed_batch)[:, :, :, ::-1].transpose(0, 3, 1, 2)
    )
    preprocessed_batch = torch.from_numpy(rgb_nchw).float().to(device) / 255.0

    return (
        preprocessed_batch,
        offset[0],
        scale[0],
    )  # Input image size should not change


def get_inference_output_shape(
    input_shape: typing.Tuple[int, int], n_classes: int
) -> typing.Tuple[int, int]:
    """Get the inference output shape given the image input shape and number of classes
    For more information regarding output shape, see the following github issue:
    https://github.com/ultralytics/yolov5/issues/1277

    Args:
        input_shape (typing.Tuple): image input shape, (h, w)
        n_classes (int): number of classes detector is trained on to predict

    Returns:
        typing.Tuple: Yolo returns an output of (N, A, V), where N is batch size, A is the number
            of anchors (determined by image input shape), and V is the prediction vector for a
            given anchor. This function returns a tuple containing (A, V)
    """
    k_anchor_points_per_grid = 3
    k_default_output_size = 5
    stride_grid_sizes = (
        (input_shape[0] / 8) * (input_shape[1] / 8),
        (input_shape[0] / 16) * (input_shape[1] / 16),
        (input_shape[0] / 32) * (input_shape[1] / 32),
    )
    n_anchors = (
        k_anchor_points_per_grid * int(stride_grid_sizes[0])
        + k_anchor_points_per_grid * int(stride_grid_sizes[1])
        + k_anchor_points_per_grid * int(stride_grid_sizes[2])
    )
    n_anchor_points = k_default_output_size + n_classes
    return (n_anchors, n_anchor_points)


def transform_detections(
    detections: torch.Tensor, offset: tuple, scale: tuple
) -> torch.Tensor:
    """transform_detections.

    this transforms the output of the detector into the resized image
    since inference was done on the resized image


    Args:
        detections (torch.Tensor): this is the raw detection output
        offset (tuple): the offset (x, y) in pixels of the detection
        scale (tuple): the scale of the resized image (x, y)

    Returns:
        torch.Tensor: the raw tensor output with the resized detections
    """
    # we just apply the transformation to the bounding box
    # transformations happen in place
    offset_x, offset_y = offset
    scale_x, scale_y = scale
    detections[:, :, 0] -= offset_x
    detections[:, :, 0] /= scale_x
    detections[:, :, 2] /= scale_x

    detections[:, :, 1] -= offset_y
    detections[:, :, 1] /= scale_y
    detections[:, :, 3] /= scale_y
    return detections


def post_process_prediction(
    predictions: torch.Tensor,
    classes: dict,
    confidence_threshold: float,
    nms_threshold: float,
) -> list:
    """post_process_prediction

    from a raw input tensor, this produces the class indexed set of bounding
    box predictions with their confidences

    Args:
        predictions (torch.Tensor): raw output tensor of the yolo model
        classes (dict): classes dictionary containing class_id and actor_category
        confidence_threshold (float): confidence threshold for byte_track postprocessing
        nms_threshold (float): nms threshold for byte_track postprocessing

    Returns:
        list: list of dictionary indexed by the class label ordered by batched images
    """
    inference_dimension = 2
    bounding_box_dim = 4
    n_inference = predictions.size()[inference_dimension]
    n_classes = n_inference - bounding_box_dim - 1
    _, _, class_confidences = torch.split(
        predictions, [bounding_box_dim, 1, n_classes], 2
    )
    class_labels = torch.argmax(class_confidences, inference_dimension)
    output_observations = []

    for i, image_prediction in enumerate(predictions):

        def class_is_predicted(index: int) -> bool:
            """Deterimine if class is predicted
            Args:
                index (int): index of prediction
            Returns:
                bool: if index prediction is an actual class
            """
            return index < n_classes

        image_observations = {}
        for class_id, actor_category in classes.items():
            if not class_is_predicted(class_id):
                continue

            class_prediction = image_prediction[class_labels[i] == class_id]
            # there is an issue with the way bytetrack uses it's detection probability
            # so we just dump the correct one in
            class_prediction[:, 5] = class_prediction[:, 5 + class_id]
            values = class_prediction[:, :6]
            post_processed = postprocess(
                values.view((1, values.size()[0], values.size()[1])),
                1,
                confidence_threshold,
                nms_threshold,
            )
            observation = post_processed[0]
            #  make the observation an empty tensor if no bounding boxes come through
            if observation is None:
                observation = torch.empty(0, n_inference)
            image_observations[actor_category] = observation

        output_observations.append(image_observations)

    return output_observations
