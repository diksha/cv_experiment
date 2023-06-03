#
# Copyright 2023 Voxel Labs, Inc.
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

import torch
import torchvision

from core.structs.actor import ActorCategory


def unpack_observations(
    observation: torch.tensor,
    num_observations: torch.tensor,
    classes: typing.Dict[int, ActorCategory],
) -> typing.List[typing.Dict[str, torch.Tensor]]:
    """
    Unpacks the observations into a list of dictionaries of detections by class
    takes a padded tensor of observations (from pad_observations),
    a tensor of the number of observations (number of valid slices in the observation tensor)
    and a dictionary of classes and returns a list of dictionary of detections by class,
    where the index is the batch index the observation tensor has shape
    (batch_size, max_num_observations, 1 + n_inference) the num_observations tensor
    has a size (batch_size, 1)

    Args:
        observation (torch.tensor): the padded observation tensor
        num_observations (torch.tensor): the number of observations tensor
        classes (typing.Dict[int, ActorCategory]): the classes dictionary

    Returns:
        typing.List[typing.Dict[str, torch.Tensor]]: a list of dictionaries of detections by class
    """
    # implements the pattern above
    results = []
    for num_observation_batch, observation_batch in zip(
        num_observations, observation
    ):
        n_batch = num_observation_batch.item()
        observation = observation_batch[:n_batch, :]
        class_ids, detections = torch.split(observation, [1, 7], dim=1)
        class_ids = class_ids.to(torch.int16).squeeze(1)
        results.append(
            {
                class_name: torch.empty((0, 7))
                for class_id, class_name in classes.items()
            }
        )
        for class_id, class_name in classes.items():
            class_mask = class_ids == class_id
            results[-1][class_name] = detections[class_mask, :]
    return results


@torch.jit.script
def get_detections(
    image_pred: torch.Tensor, confidence_threshold: float, num_classes: int
) -> torch.Tensor:
    """
    Generates detections with the confidence threshold
    to be used downstream in bytetrack and NMS

    Note: derived from third_party/bytetrack/utils.py

    Args:
        image_pred (torch.Tensor): the raw prediction from the model
        confidence_threshold (float): the confidence threshold to filter
        num_classes (int): the number of classes to use

    Returns:
        torch.Tensor: the detections
    """
    class_score_dim = 5
    # Get score and class with highest confidence
    class_conf, class_pred = torch.max(
        image_pred[:, class_score_dim : class_score_dim + num_classes],
        1,
        keepdim=True,
    )

    class_probability_dim = 4

    conf_mask = (
        image_pred[:, class_probability_dim] * class_conf.squeeze()
        >= confidence_threshold
    ).squeeze()
    # Detections structured as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
    detections = torch.cat(
        (image_pred[:, :class_score_dim], class_conf, class_pred.float()), 1
    )
    detections = detections[conf_mask]
    return detections


@torch.jit.script
def postprocess(
    prediction: torch.Tensor,
    num_classes: int,
    confidence_threshold: float = 0.7,
    nms_thre: float = 0.45,
) -> typing.List[typing.Optional[torch.Tensor]]:
    """
    Postprocesses the output of the model to filter out low confidence
    detections and to perform NMS

    Args:
        prediction (torch.Tensor): the raw prediction from the model
        num_classes (int): the number of classes
        confidence_threshold (float, optional): The confidence threshold. Defaults to 0.7.
        nms_thre (float, optional): the NMS threshold. Defaults to 0.45.

    Returns:
        typing.List[typing.Optional[torch.Tensor]]: a list of detections if they exist
    """

    box_corner = torch.ones(prediction.shape, dtype=prediction.dtype)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output: typing.List[typing.Optional[torch.Tensor]] = [
        None for _ in range(len(prediction))
    ]
    image_pred = prediction[0]

    # If none are remaining => process next image
    if not image_pred.size(0) > 0:
        return output
    detections = get_detections(image_pred, confidence_threshold, num_classes)
    if not detections.size(0) > 0:
        return output

    nms_out_index = torchvision.ops.batched_nms(
        detections[:, :4],
        detections[:, 4] * detections[:, 5],
        detections[:, 6],
        nms_thre,
    )
    detections = detections[nms_out_index]
    output[0] = detections

    return output


@torch.jit.script
def transform_detections(
    detections: torch.Tensor,
    offset: typing.Tuple[float, float],
    scale: typing.Tuple[float, float],
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


@torch.jit.script
def class_is_predicted(index: int, n_classes: int) -> bool:
    """Deterimine if class is predicted
    Args:
        index (int): index of prediction
        n_classes (int): the number of classes the detector is trained on
    Returns:
        bool: if index prediction is an actual class
    """
    return index < n_classes


@torch.jit.script
def generate_preprocess_values(
    image_prediction: torch.Tensor,
    class_labels: torch.Tensor,
    index: int,
    class_id: int,
) -> torch.Tensor:
    """
    Helper function to generate values amenable for postprocessing

    Args:
        image_prediction (torch.Tensor): the image prediction
        class_labels (torch.Tensor): the class labels for the image
        index (int): the index of the batch to process
        class_id (int): the class id to process

    Returns:
        torch.Tensor: the values to process
    """
    class_prediction = image_prediction[class_labels[index] == class_id]
    # there is an issue with the way bytetrack uses it's detection probability
    # so we just dump the correct one in
    class_prediction[:, 5] = class_prediction[:, 5 + class_id]
    values = class_prediction[:, :6]
    return values.view((1, values.size()[0], values.size()[1]))


@torch.jit.script
def _apply_post_processing_to_index(
    image_prediction: torch.Tensor,
    class_labels: torch.Tensor,
    index: int,
    class_id: int,
    confidence_threshold: float,
    nms_threshold: float,
    n_inference: int,
) -> torch.Tensor:
    """
    Takes the individual image prediction and applies
    NMS confidence thresholding, and inference thresholding
    to the index of the batch

    Args:
        image_prediction (torch.Tensor): current image prediction
        class_labels (torch.Tensor): the class labels for the image
        index (int): the index of the batch to process
        class_id (int): the class id to process
        confidence_threshold (float): the confidence threshold to apply
        nms_threshold (float): the nms threshold to apply
        n_inference (int): the number inferences present in the prediction tensor

    Returns:
        torch.Tensor: the post processed tensor
    """
    values = generate_preprocess_values(
        image_prediction=image_prediction,
        class_labels=class_labels,
        index=index,
        class_id=class_id,
    )
    post_processed: typing.List[typing.Optional[torch.Tensor]] = postprocess(
        values,
        1,
        confidence_threshold,
        nms_threshold,
    )
    observation = post_processed[0]
    if observation is None:
        observation = torch.empty(0, n_inference)
    return observation


@torch.jit.script
def pad_observations(
    observation_list: typing.List[torch.Tensor],
) -> torch.Tensor:
    """
    Pad the list of observations to the max length
    Args:
        observation_list (typing.List[torch.Tensor]): list of observations
    Returns:
        torch.Tensor: the padded tensor
    """
    max_length = max(observation.size()[0] for observation in observation_list)
    # the output must be at least 1 or torch throws an error
    max_length = max(max_length, 1)
    padded = torch.zeros(
        (len(observation_list), max_length, observation_list[0].size()[1])
    )
    for i, observation in enumerate(observation_list):
        padded[i, : observation.size()[0], :] = observation
    return padded


@torch.jit.script
def post_process_prediction(
    predictions: torch.Tensor,
    classes: torch.Tensor,
    confidence_threshold: float,
    nms_threshold: float,
) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    """post_process_prediction

    from a raw input tensor, this produces the class indexed set of bounding
    box predictions with their confidences

    Args:
        predictions (torch.Tensor): raw output tensor of the yolo model
        classes (typing.List[int]): the class indexes to use
        confidence_threshold (float): confidence threshold for byte_track postprocessing
        nms_threshold (float): nms threshold for byte_track postprocessing

    Returns:
        typing.Tuple[torch.Tensor, torch.Tensor]: tuple containing
            tensor index 0: has a size (batch_size, 1) and corresponds to the number of observations
            tensor index 1: (batch_size, max_num_observations, 1 + n_inference) the
                       padded postprocess tensor
    """
    inference_dimension = 2
    bounding_box_dim = 4
    n_inference = predictions.size()[inference_dimension]
    n_classes = n_inference - bounding_box_dim - 1
    _, _, class_confidences = torch.split(
        predictions, [bounding_box_dim, 1, n_classes], 2
    )
    class_labels = torch.argmax(class_confidences, inference_dimension)
    output_observations: typing.List[torch.Tensor] = []
    output_counts: torch.Tensor = torch.zeros(
        len(predictions), dtype=torch.int64
    )
    output_inference_size = n_inference + 1

    for i, image_prediction in enumerate(predictions):
        image_observations = torch.empty(
            (0, output_inference_size), device=predictions.device
        )
        counts = 0
        for class_id in classes:
            if not class_is_predicted(class_id, n_classes):
                continue

            observation = _apply_post_processing_to_index(
                image_prediction=image_prediction,
                class_labels=class_labels,
                index=i,
                class_id=class_id,
                confidence_threshold=confidence_threshold,
                nms_threshold=nms_threshold,
                n_inference=n_inference,
            )
            counts += observation.size()[0]

            class_index_t = class_id * torch.ones(
                (observation.size()[0], 1), device=predictions.device
            )
            if observation.size()[0] > 0:
                #  make the observation an empty tensor if no bounding boxes come through
                image_observation = torch.cat(
                    [class_index_t, observation],
                    dim=1,
                )
            else:
                image_observation = torch.empty(
                    0, output_inference_size, device=predictions.device
                )
            image_observations = torch.cat(
                [image_observations, image_observation], dim=0
            )
        output_counts[i] = counts
        output_observations.append(image_observations)
    if output_observations is None:
        output_observations = torch.empty(
            0, output_inference_size, device=predictions.device
        )

    return output_counts, pad_observations(output_observations)


@torch.jit.script
def transform_and_post_process(
    prediction: torch.Tensor,
    offset: torch.Tensor,
    scale: torch.Tensor,
    classes: torch.Tensor,
    confidence_threshold: torch.Tensor,
    nms_threshold: torch.Tensor,
) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    """
    Transforms and postprocesses
    the raw prediction from the model
    so the results can be used downstream in the tracker

    NOTE: this does not take the batch into account for offset, nms,
    confidence threshold, and scale. For remote execution (i.e. triton)
    ensure that the max batchsize is one to eliminate incompatibility

    Args:
        prediction (torch.Tensor): the raw prediction from the model
        offset (typing.Tuple[float, float]): the offset of the image
        scale (typing.Tuple[float, float]): the scale of the image
        classes (typing.List[int]): the classes to use
        confidence_threshold (float): the confidence threshold to apply
        nms_threshold (float): the nms threshold to apply

    Returns:
        list: the postprocessed observations in format:
        [(class_label_index, output_tensor)... for batch in batchsize]
    """
    transformed_prediction = transform_detections(
        prediction, (offset[0][0], offset[0][1]), (scale[0][0], scale[0][1])
    )
    observations = post_process_prediction(
        transformed_prediction,
        classes=classes[0],
        confidence_threshold=confidence_threshold[0][0],
        nms_threshold=nms_threshold[0][0],
    )
    return observations
