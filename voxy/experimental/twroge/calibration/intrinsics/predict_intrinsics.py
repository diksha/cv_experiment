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
# adapted from: https://github.com/alexvbogdan/DeepCalib

# trunk-ignore-all(pylint)
# trunk-ignore-all(flake8)

from __future__ import print_function

import argparse
import glob
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys

import cv2
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import optimizers
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.models import Model

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

PATH_TO_WEIGHTS = "artifacts_deepcalib_pretrained_weights/weights_10_0.02.h5"
IMAGE_SIZE = 299
INPUT_SIZE = 299


def get_model() -> Model:
    """get_model.

    Gets the deepcalib model using the pretrained weights

    Args:

    Returns:
        Model: the single net model to predict intrinsic and distortion
    """
    input_shape = (299, 299, 3)
    main_input = Input(shape=input_shape, dtype="float32", name="main_input")
    phi_model = InceptionV3(
        weights="imagenet",
        include_top=False,
        input_tensor=main_input,
        input_shape=input_shape,
    )
    phi_features = phi_model.output
    phi_flattened = Flatten(name="phi-flattened")(phi_features)
    final_output_focal = Dense(1, activation="sigmoid", name="output_focal")(
        phi_flattened
    )
    final_output_distortion = Dense(
        1, activation="sigmoid", name="output_distortion"
    )(phi_flattened)

    layer_index = 0

    model = Model(
        inputs=main_input,
        outputs=[final_output_focal, final_output_distortion],
    )
    model.load_weights(PATH_TO_WEIGHTS)
    return model


def crop_from_center(image):
    x = image.shape[1] // 2 - INPUT_SIZE // 2
    y = image.shape[0] // 2 - INPUT_SIZE // 2

    crop_image = image[int(y) : int(y + 299), int(x) : int(x + 299)]
    return crop_image


def resize_from_center(image):
    final_size = INPUT_SIZE
    ratio = float(image.shape[0]) / float(image.shape[1])
    if ratio < 1:
        width = int(final_size / ratio)
        resized = cv2.resize(
            image,
            (int(final_size / ratio), final_size),
            interpolation=cv2.INTER_LINEAR,
        )
    else:
        height = int(final_size * ratio)
        resized = cv2.resize(
            image,
            (final_size, int(final_size * ratio)),
            interpolation=cv2.INTER_LINEAR,
        )
    resized = crop_from_center(resized)
    return resized


def preprocess_image(image: np.array) -> np.array:
    """preprocess_image.

    This preprocesses the image for the deepcalib model

    Args:
        image (np.array): image the raw image as loaded from imread

    Returns:
        np.array: normalized and resized image
    """
    image = resize_from_center(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0
    image = image - 0.5
    image = image * 2.0
    image = np.expand_dims(image, 0)
    image = preprocess_input(image)
    return image


def scale_model_output(prediction: np.array) -> tuple:
    """scale_model_output.

    This scales the model output from the normalized network output to the full
    range given by the way the model was trained

    Args:
        prediction (np.array): the raw output of the model from an image

    Returns:
        tuple: the correctly scaled focal length and distortion
    """
    FOCAL_START = 40
    FOCAL_END = 500
    focal_length_prediction = prediction[0]
    distortion_prediction = prediction[1]
    focal_length_prediction_pixels = (
        (
            focal_length_prediction[0][0]
            * (FOCAL_END + 1.0 - FOCAL_START * 1.0)
            + FOCAL_START * 1.0
        )
        * (IMAGE_SIZE * 1.0)
        / (INPUT_SIZE * 1.0)
    )
    distortion_prediction_xi = distortion_prediction[0][0] * 1.2
    return focal_length_prediction_pixels, distortion_prediction_xi


def get_paths(IMAGE_FILE_PATH_DISTORTED):
    """get_paths.

    Gets all the jpgs and pngs from a directory and outputs it as a list

    Args:
        IMAGE_FILE_PATH_DISTORTED: the file path with raw distorted images
    """

    image_paths = glob.glob(os.path.join(IMAGE_FILE_PATH_DISTORTED, "*.jpg"))
    image_paths.extend(
        glob.glob(os.path.join(IMAGE_FILE_PATH_DISTORTED, "*.png"))
    )
    image_paths.sort()
    parameters = []
    labels_focal_test = []

    return image_paths


def main(args):
    filename_results = os.path.join(args.output, "results.txt")
    IMAGE_FILE_PATH_DISTORTED = args.image_directory

    print(f"Output file: {filename_results} ")

    image_paths = get_paths(IMAGE_FILE_PATH_DISTORTED)
    model = get_model()

    with tf.device("/gpu:0"):

        with open(filename_results, "w") as file:
            for path in image_paths:
                image = cv2.imread(path)
                image = preprocess_image(image)

                prediction = model.predict(image)
                (
                    focal_length_prediction_pixels,
                    distortion_prediction_xi,
                ) = scale_model_output(prediction)
                prediction_string = f"{path}\tfocal_length_prediction_pixels\t{focal_length_prediction_pixels}\tprediction_distortion\t{distortion_prediction_xi}\n"
                file.write(prediction_string)
                print(prediction_string, end="")


def get_args():
    parser = argparse.ArgumentParser(description="Predict Intrinsics")
    parser.add_argument(
        "--image_directory",
        type=str,
        required=True,
        help="The image directory for ",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="The output directory to put the model",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    main(args)
