#
# Copyright 2022 Voxel Labs, Inc.
# All rights reserved.
#
# This document may not be reproduced, republished, distributed, transmitted,
# displayed, broadcast or otherwise exploited in any manner without the express
# prior written permission of Voxel Labs, Inc. The receipt or possession of this
# document does not convey any rights to reproduce, disclose, or distribute its
# contents, or to manufacture, use, or sell anything that it may describe, in
# whole or in part.
#

# Example run:  ./bazel run core/labeling/scale/runners:camera_config_scale_task --
# --camera_uuid innovate_manufacturing/knoxville/0004/cha
import argparse
import json
import os
import tempfile
from typing import Callable, List, Tuple

import shortuuid
import xmltodict
from loguru import logger
from scaleapi.tasks import TaskType

from core.labeling.cvat.client import CVATClient
from core.labeling.scale.lib.scale_client import get_scale_client
from core.labeling.scale.lib.scale_task_retry import ScaleTaskRetryWrapper
from core.labeling.scale.lib.scale_task_wrapper import ScaleTaskWrapper
from core.utils.aws_utils import (
    get_secret_from_aws_secret_manager,
    upload_file,
)


def get_annotations_from_cvat(camera_uuid) -> Tuple[List, str]:
    """Get annotations and s3 path of image from cvat

    Args:
        camera_uuid (str): uuid of the camera

    Returns:
        Tuple[List,str]: list of annotations and s3path
    """

    def get_points(points) -> List:
        """Get vertices from points

        Args:
            points (list): list of points

        Returns:
            List: list of vertices x,y
        """
        vertices = []
        for ele in points.split(";"):
            x_point, y_point = ele.split(",")
            vertices.append({"x": float(x_point), "y": float(y_point)})
        return vertices

    def get_scale_annotation(polygon) -> dict:
        """Given polygon get scale annotations

        Args:
            polygon (dict): polygon dictionary

        Returns:
            dict: annotations in scale format
        """
        annotation = {}
        annotation["label"] = polygon["@label"]
        annotation["vertices"] = get_points(polygon["@points"])
        attributes = {}
        cvat_attributes = polygon.get("attribute")
        if cvat_attributes:
            if isinstance(cvat_attributes, dict):
                attributes[cvat_attributes["@name"].lower()] = cvat_attributes[
                    "#text"
                ]
            else:
                for attribute in cvat_attributes:
                    if attribute["#text"].isdigit():
                        attributes[attribute["@name"].lower()] = int(
                            attribute["#text"]
                        )
                    else:
                        attributes[attribute["@name"].lower()] = attribute[
                            "#text"
                        ]
        annotation["attributes"] = attributes
        annotation["type"] = "polygon"
        return annotation

    cvat_client = CVATClient(
        "cvat.voxelplatform.com",
        get_secret_from_aws_secret_manager("CVAT_CREDENTIALS"),
        project_id=10,
    )
    task_name = f"camera_config/{camera_uuid}"
    with tempfile.NamedTemporaryFile() as temp:
        cvat_client.download_task_data_camera_config(task_name, temp.name)
        upload_file(
            bucket="voxel-logs",
            local_path=temp.name,
            s3_path=f"{task_name}.png",
        )

    with tempfile.TemporaryDirectory() as tmpdirname:
        cvat_client.download_cvat_labels(
            task_name, "cvat_images", False, tmpdirname
        )
        with open(
            os.path.join(tmpdirname, task_name, "annotations.xml"),
            encoding="UTF-8",
        ) as in_file:
            annotations = []
            data_dict = xmltodict.parse(in_file.read())
            polygons = data_dict["annotations"]["image"].get("polygon", None)
            if polygons:
                if isinstance(polygons, dict):
                    polygon = polygons
                    annotations.append(get_scale_annotation(polygon))
                else:
                    for polygon in polygons:
                        annotations.append(get_scale_annotation(polygon))
            return annotations, f"s3://voxel-logs/{task_name}.png"


class CameraConfigAnnotationTask:
    _TAXONOMY_PATH = "core/labeling/scale/task_creation/taxonomies"

    def __init__(self, camera_uuid: str, credentials_arn: str):
        self.project = "camera_config"
        self.camera_uuid = camera_uuid
        self.client = get_scale_client(credentials_arn)
        taxonomy_path = os.path.join(
            self._TAXONOMY_PATH, f"{self.project}.json"
        )
        self.credentials_arn = credentials_arn
        with open(taxonomy_path, "r", encoding="UTF-8") as taxonomy_file:
            self.taxonomy = json.load(taxonomy_file)
        self.batch = self.client.create_batch(
            project=self.project,
            batch_name=f"batch_{self.camera_uuid}_{shortuuid.uuid()}",
        )

    def create_task(self) -> None:
        """Create a task for camera config

        Raises:
            RuntimeError: task creation failed
        """
        annotations, s3_path = get_annotations_from_cvat(self.camera_uuid)
        logger.info(f"Image path {s3_path}")
        camera_uuid = self.camera_uuid
        try:
            payload = dict(
                project=self.project,
                batch=self.batch.name,
                attachment=s3_path,
                metadata={
                    "camera_uuid": camera_uuid,
                    "filename": camera_uuid,
                },
                unique_id=f"{camera_uuid}",
                clear_unique_id_on_error=True,
                geometries=self.taxonomy["geometries"],
                annotation_attributes=self.taxonomy["annotation_attributes"],
                hypothesis={"annotations": annotations},
            )

            def create_task() -> Callable:
                """Create task function for camera_config

                Returns:
                    Callable: create scale task function
                """
                return self.client.create_task(
                    TaskType.ImageAnnotation,
                    **payload,
                )

            def cancel_task() -> Callable:
                """Cancel scale task function for camera config

                Returns:
                    Callable: Cancel scale task callable
                """
                return self.client.cancel_task(
                    ScaleTaskWrapper(
                        self.credentials_arn
                    ).get_task_id_from_unique_id(
                        camera_uuid,
                        self.project,
                    ),
                    True,
                )

            ScaleTaskRetryWrapper(
                task_creation_call=create_task,
                task_cancel_call=cancel_task,
            ).create_task()
        except Exception:  # trunk-ignore(pylint/W0703)
            logger.exception(f"Failed to create task for {camera_uuid}")
        self.batch.finalize()
        logger.info("create scale task")


def parse_args() -> argparse.Namespace:
    """Parses arguments for the binary

    Returns:
        argparse.Namespace: Command line arguments passed in.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--camera_uuid",
        type=str,
        help="camera_uuid",
    )
    parser.add_argument(
        "-a",
        "--credentials_arn",
        type=str,
        default=(
            "arn:aws:secretsmanager:us-west-2:203670452561:"
            "secret:scale_credentials-WHUbar"
        ),
        help="Credetials arn",
    )
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    CameraConfigAnnotationTask(
        args.camera_uuid, args.credentials_arn
    ).create_task()
