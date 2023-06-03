import argparse
import logging
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml

from core.perception.calibration.camera_model import CameraModel
from core.perception.calibration.utils import (
    calibration_config_to_camera_model,
)

logging.getLogger().setLevel(logging.INFO)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Calibration")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="The path to an image with aruco targets",
    )
    parser.add_argument(
        "--calibration-file",
        type=str,
        required=True,
        help="The calibration file in yaml format",
    )
    parser.add_argument(
        "--tag-size",
        type=float,
        required=True,
        help="The tag size of the aruco target in meters",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="The output directory to put the validation plots",
    )
    return parser.parse_args()


def read_calibration(filename: str) -> CameraModel:
    with open(filename, "r", encoding="utf-8") as calibration_file:
        calibration_config = yaml.safe_load(calibration_file)
        return calibration_config_to_camera_model(calibration_config)


def get_keypoints(detection: np.array) -> zip:
    corner_loop = []
    corner_loop.extend(detection)
    corner_loop.append(detection[0])
    return zip(corner_loop[:-1], corner_loop[1:])


def generate_statistics(distances: list, percent_errors: list) -> None:
    mu_distances = np.mean(distances)
    median_distances = np.median(distances)
    sigma_distances = np.std(distances)
    distance_log = f"Distances: \tmean: {mu_distances} median: {median_distances} std: {sigma_distances} "

    mu_percent_error = np.mean(percent_errors)
    median_percent_error = np.median(percent_errors)
    sigma_percent_error = np.std(percent_errors)
    error_log = f"% Errors: \tmean: {mu_percent_error} median: {median_percent_error} std: {sigma_percent_error} "
    logging.info("-" * len(distance_log))
    logging.info(distance_log)
    logging.info(error_log)
    logging.info("-" * len(distance_log))


def generate_plots(
    output: str,
    name: str,
    xaxis: str,
    yaxis: str,
    values: list,
    true_val: float,
) -> None:
    # generate histogram
    plt.figure()
    print(name)
    print(values)
    plt.hist(values, 10, density=True, facecolor="g", alpha=0.75)
    plt.grid(True)
    plt.title(name)
    plt.xlabel(xaxis)
    plt.ylabel(yaxis)
    if true_val is not None:
        plt.axvline(x=true_val, label="True value")
        plt.legend()
    # generate plots
    plt.savefig(os.path.join(output, name.lower().replace(" ", "_") + ".png"))


def main(args: argparse.Namespace) -> None:
    logging.info("[Validating calibration]")
    logging.info(f"Image: {args.image}")
    logging.info(f"Calibration file: {args.calibration_file}")
    logging.info(f"Tag size: {args.tag_size}")

    camera_model = read_calibration(args.calibration_file)
    image = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detections, ids, _ = cv2.aruco.detectMarkers(image, dictionary)

    percent_errors = []
    distances = []
    distance_errors = []
    for detection in detections:
        for point_a, point_b in get_keypoints(detection[0]):
            x_a, y_a = camera_model.project_image_point(point_a[0], point_a[1])
            x_b, y_b = camera_model.project_image_point(point_b[0], point_b[1])
            distance = np.linalg.norm(np.array([x_a - x_b, y_a - y_b]))
            true_distance = args.tag_size
            distance_error = abs(true_distance - distance)
            distance_errors.append(distance_error)
            distances.append(distance)
            logging.info(f"Distance: {distance}")
            percent_error = (
                abs((true_distance - distance) / true_distance) * 100
            )
            percent_errors.append(percent_error)
            logging.info(f"Percent error: {percent_error} ")
    # plot histogram of distances, errors
    generate_statistics(distances, percent_errors)
    generate_plots(
        args.output,
        "Percent Error distribution",
        "Percent Error",
        "density (%)",
        percent_errors,
        None,
    )
    generate_plots(
        args.output,
        "Distance Error distribution",
        "Distance Error (m)",
        "density (%)",
        distance_errors,
        None,
    )
    generate_plots(
        args.output,
        "Distances distribution",
        "Distances (m)",
        "density (%)",
        distances,
        true_distance,
    )


if __name__ == "__main__":
    arguments = get_args()
    main(arguments)
