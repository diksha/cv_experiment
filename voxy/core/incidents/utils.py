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

import copy
import importlib
import inspect
import json
from dataclasses import dataclass

from core.incidents import monitors
from core.incidents.monitors.base import MonitorBase
from core.structs.attributes import Line, Point, Polygon

_CAMERA_CONFIG_PATH = "configs/cameras/camera_config.json"
_CAMERA_CONFIG = None


def _camera_config() -> dict:
    # trunk-ignore(pylint/W0603)
    global _CAMERA_CONFIG
    if _CAMERA_CONFIG is None:
        with open(
            _CAMERA_CONFIG_PATH, "r", encoding="UTF-8"
        ) as camera_config_file:
            _CAMERA_CONFIG = json.loads(camera_config_file.read())
    return _CAMERA_CONFIG


@dataclass
class DoorCameraConfig:
    polygon: Polygon
    orientation: str
    door_id: str
    door_type: str


@dataclass
class MotionZoneCameraConfig:
    polygon: Polygon
    zone_id: int


def is_monitor(cls):
    """Is subclass of base class, but not the actual base class."""
    if not inspect.isclass(cls):
        return False
    return issubclass(cls, MonitorBase) and cls != MonitorBase


def iter_monitors(monitors_requested=["all"], hot_reload=False):
    """Iterates through monitor classes defined in monitor directory."""
    all_selected = "all" in monitors_requested
    for module_name in monitors.__all__:
        relative_module_name = ".{}".format(module_name)
        module = importlib.import_module(
            relative_module_name, "core.incidents.monitors"
        )
        if hot_reload:
            importlib.reload(module)
        for _, monitor in inspect.getmembers(module, is_monitor):
            if all_selected or monitor.NAME in monitors_requested:
                yield monitor


class CameraConfigError(RuntimeError):
    pass


class CameraConfig:  # trunk-ignore(pylint/R0902)
    """
    Camera config wrapper
    """

    def __init__(self, camera_uuid, frame_height, frame_width):
        self.doors = []
        self.driving_areas = []
        self.actionable_regions = []
        self.intersections = []
        self.aisle_ends = []
        self.no_pedestrian_zones = []
        self.motion_detection_zones = []
        self.no_obstruction_regions = []
        self.next_door_id = 1
        camera_config = _camera_config()

        if not camera_config:
            raise RuntimeError("Camera config file not found")
        if camera_uuid not in camera_config:
            raise CameraConfigError(
                f"Camera uuid {camera_uuid} not in camera config"
            )
        self.config_dict = copy.deepcopy(camera_config[camera_uuid])
        for door in self.config_dict.get("doors", []):
            self.doors.append(
                DoorCameraConfig(
                    polygon=self._unnormalize_polygon(
                        Polygon(
                            vertices=[
                                Point(item[0], item[1])
                                for item in door["polygon"]
                            ]
                        ),
                        frame_height,
                        frame_width,
                    ),
                    orientation=door["orientation"].upper(),
                    door_id=door["door_id"],
                    door_type=door["type"].upper(),
                )
            )
        for area in self.config_dict.get("drivingAreas", []):
            self.driving_areas.extend(
                [
                    self._unnormalize_polygon(
                        Polygon(
                            vertices=[
                                Point(item[0], item[1])
                                for item in area["polygon"]
                            ]
                        ),
                        frame_height,
                        frame_width,
                    )
                ]
            )
        for region in self.config_dict.get("actionableRegions", []):
            self.actionable_regions.extend(
                [
                    self._unnormalize_polygon(
                        Polygon(
                            vertices=[
                                Point(item[0], item[1])
                                for item in region["polygon"]
                            ]
                        ),
                        frame_height,
                        frame_width,
                    )
                ]
            )
        for region in self.config_dict.get("intersections", []):
            self.intersections.extend(
                [
                    self._unnormalize_polygon(
                        Polygon(
                            vertices=[
                                Point(item[0], item[1])
                                for item in region["polygon"]
                            ]
                        ),
                        frame_height,
                        frame_width,
                    )
                ]
            )
        for aisle_end in self.config_dict.get("endOfAisles", []):
            self.aisle_ends.extend(
                [
                    self._unnormalize_line(
                        Line(
                            points=[
                                Point(item[0], item[1])
                                for item in aisle_end["polygon"]
                            ]
                        ),
                        frame_height,
                        frame_width,
                    )
                ]
            )
        for region in self.config_dict.get("noPedestrianZones", []):
            self.no_pedestrian_zones.extend(
                [
                    self._unnormalize_polygon(
                        Polygon(
                            vertices=[
                                Point(item[0], item[1])
                                for item in region["polygon"]
                            ]
                        ),
                        frame_height,
                        frame_width,
                    )
                ]
            )
        for region in self.config_dict.get("noObstructionRegions", []):
            self.no_obstruction_regions.extend(
                [
                    self._unnormalize_polygon(
                        Polygon(
                            vertices=[
                                Point(item[0], item[1])
                                for item in region["polygon"]
                            ]
                        ),
                        frame_height,
                        frame_width,
                    )
                ]
            )
        for zone in self.config_dict.get("motionDetectionZones", []):
            self.motion_detection_zones.append(
                MotionZoneCameraConfig(
                    polygon=self._unnormalize_polygon(
                        Polygon(
                            vertices=[
                                Point(item[0], item[1])
                                for item in zone["polygon"]
                            ]
                        ),
                        frame_height,
                        frame_width,
                    ),
                    zone_id=zone["zone_id"],
                )
            )
        self.next_door_id = self.config_dict.get("nextDoorId", 1)
        self.version = self.config_dict.get("version")

    def _unnormalize_polygon(
        self, normalized_polygon, frame_height, frame_width
    ):
        return Polygon(
            vertices=[
                Point(
                    int(vertex.x * frame_width), int(vertex.y * frame_height)
                )
                for vertex in normalized_polygon.vertices
            ]
        )

    def _unnormalize_line(self, line, frame_height, frame_width):
        for point in line.points:
            point.x = point.x * frame_width
            point.y = point.y * frame_height
        return line

    def to_dict(self) -> dict:
        """Dictionary of camera config values

        Returns:
            dict: Dictionary of camera config values
        """
        return self.config_dict
