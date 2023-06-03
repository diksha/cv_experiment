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
import json
from typing import List

import attr
import numpy as np
from foxglove_schemas_protobuf.CircleAnnotation_pb2 import CircleAnnotation
from foxglove_schemas_protobuf.Color_pb2 import Color
from foxglove_schemas_protobuf.ImageAnnotations_pb2 import ImageAnnotations
from foxglove_schemas_protobuf.Point2_pb2 import Point2
from foxglove_schemas_protobuf.PointsAnnotation_pb2 import PointsAnnotation
from shapely.geometry import LineString as ShapelyLineString
from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import Polygon as ShapelyPolygon

from core.common.utils.proto_utils import VoxelProto

# trunk-ignore-begin(pylint/E0611)
from protos.perception.types.v1.types_pb2 import KeyPoint as KeyPointPb
from protos.perception.types.v1.types_pb2 import Polygon as PolygonPb
from protos.perception.types.v1.types_pb2 import Pose as PosePb

# trunk-ignore-end(pylint/E0611)
# The file is too long, TODO: we should look into reducing this boilerplate
# trunk-ignore-all(pylint/C0302)


# TODO (Anurag): Figure out how to fix Trunk errors with attr, slots, and __init__
@attr.s(slots=True)
class Point:

    x = attr.ib(type=float)
    y = attr.ib(type=float)
    z = attr.ib(default=None, type=float)

    def __init__(self, x: float, y: float, z: float = None) -> None:
        self.x = x
        self.y = y
        self.z = z

    def to_dict(self):
        return {"x": self.x, "y": self.y, "z": self.z}

    @classmethod
    def from_dict(cls, data):
        return Point(x=data["x"], y=data["y"], z=data.get("z"))

    def to_shapely_point(self):
        return ShapelyPoint(self.x, self.y)


@attr.s(slots=True)
class Line:
    points = attr.ib(type=list)

    def __init__(self, points: list):
        self.points = points

    def to_shapely_line(self):
        point_tuple_list = [(p.x, p.y) for p in self.points]
        return ShapelyLineString(point_tuple_list)

    def to_dict(self):
        return {"points": [point.to_dict() for point in self.points]}

    @classmethod
    def from_dict(cls, data):
        return Line(
            points=[Point.from_dict(point) for point in data["points"]]
        )


@attr.s(slots=True)
class KeyPoint:

    x = attr.ib(type=float)
    y = attr.ib(type=float)
    z = attr.ib(default=None, type=float)
    confidence = attr.ib(default=None, type=float)

    def __init__(
        self,
        x: float,
        y: float,
        z: float = None,
        confidence: float = None,
    ):
        self.x = x
        self.y = y
        self.z = z
        self.confidence = confidence

    def to_dict(self):
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data):
        if data is None:
            return None
        return KeyPoint(
            x=data["x"],
            y=data["y"],
            z=data.get("z"),
            confidence=data.get("confidence"),
        )

    def __add__(self, o):
        return np.array((self.x + o.x, self.y + o.y))

    def __sub__(self, o):
        return np.array((self.x - o.x, self.y - o.y))

    # return 2 dimensional version
    def to_shapely_point(self):
        return ShapelyPoint(self.x, self.y)

    def to_proto(self) -> KeyPointPb:
        """
        Generates a proto based on the keypoint

        Returns:
            KeyPointPb: the generated protobuf
        """
        kwargs = {}

        # protos do not allow assignment for optional members
        # so we have to do this..
        def safe_update(name: str, value: float):
            """
            Safely updates the keyword arguments that get passed to the
            keypoint proto constructor

            Args:
                name (str): the name of the argument
                value (float): the value of the argument
            """
            if value is not None:
                kwargs[name] = value

        safe_update("x", self.x)
        safe_update("y", self.y)
        safe_update("z", self.z)
        safe_update("confidence_probability", self.confidence)
        return KeyPointPb(**kwargs)

    @classmethod
    def from_proto(cls, raw_proto: KeyPointPb) -> "KeyPoint":
        """
        Generates a keypoint based on the protobuf definition

        Args:
            raw_proto (KeyPointPb): current protobuf to serialize from

        Returns:
            KeyPoint: the keypoint
        """
        proto = VoxelProto(raw_proto)
        return KeyPoint(
            x=proto.x,
            y=proto.y,
            z=proto.z,
            confidence=proto.confidence_probability,
        )


@attr.s(slots=True)
class Polygon:

    vertices = attr.ib(type=list)

    def __init__(self, vertices: List):
        self.vertices = vertices

    def to_dict(self):
        return {"vertices": [vertex.to_dict() for vertex in self.vertices]}

    def to_shapely_polygon(self):
        return ShapelyPolygon([[p.x, p.y] for p in self.vertices])

    def is_polygon_valid(self) -> bool:
        """Check if the shapely polygon is a valid polygon i.e., is not self-intersecting

        Returns:
            bool: True if valid else false
        """
        return ShapelyPolygon([[p.x, p.y] for p in self.vertices]).is_valid

    def get_top_left(self):
        # This only works for axis aligned rectangles.
        # Move to shapely for non axis aligned.
        X = [vertex.x for vertex in self.vertices]
        Y = [vertex.y for vertex in self.vertices]
        return Point(min(X), min(Y))

    def get_bottom_right(self):
        # This only works for axis aligned rectangles.
        # Move to shapely for non axis aligned.
        X = [vertex.x for vertex in self.vertices]
        Y = [vertex.y for vertex in self.vertices]
        return Point(max(X), max(Y))

    def get_bottom_left(self):
        # This only works for axis aligned rectangles.
        # Move to shapely for non axis aligned.
        X = [vertex.x for vertex in self.vertices]
        Y = [vertex.y for vertex in self.vertices]
        return Point(min(X), max(Y))

    def get_top_right(self):
        # This only works for axis aligned rectangles.
        # Move to shapely for non axis aligned.
        X = [vertex.x for vertex in self.vertices]
        Y = [vertex.y for vertex in self.vertices]
        return Point(max(X), min(Y))

    def get_bottom_center(self):
        """Computes the center of the bottom line of the Polygon. The Polygon is assumed to
        be aligned with the x axis and hence the y coord of bottom left and bottom right points
        of the Polygon are assumed to be same.

        Returns:
            Point: Point representing the center of the bottom of the Polygon
        """
        bottom_left = self.get_bottom_left()
        bottom_right = self.get_bottom_right()
        center_x = (bottom_left.x + bottom_right.x) / 2
        center_y = bottom_right.y
        return Point(center_x, center_y)

    def iou(self, polygon: "Polygon"):
        self_polygon = self.to_shapely_polygon()
        input_polygon = polygon.to_shapely_polygon()
        return (
            self_polygon.intersection(input_polygon).area
            / self_polygon.union(input_polygon).area
        )

    @classmethod
    def from_dict(cls, data):
        return Polygon(
            vertices=[Point.from_dict(vertex) for vertex in data["vertices"]]
        )

    @classmethod
    def from_metaverse(cls, data):
        return Polygon(
            vertices=[
                Point.from_dict(vertex)
                for vertex in json.loads(data)["vertices"]
            ]
        )

    @classmethod
    def from_bbox(cls, bbox: List) -> "Polygon":
        return Polygon(
            vertices=[
                Point(bbox[0], bbox[1]),
                Point(bbox[2], bbox[1]),
                Point(bbox[2], bbox[3]),
                Point(bbox[0], bbox[3]),
            ]
        )

    @classmethod
    def from_xysr(cls, xysr: List) -> "Polygon":
        x, y, s, r = xysr[0], xysr[1], xysr[2], xysr[3]
        w = np.sqrt(s / r)
        h = r * w
        x1 = x - w / 2.0
        x2 = x + w / 2.0
        y1 = y - h / 2.0
        y2 = y + h / 2.0
        return Polygon.from_bbox([x1, y1, x2, y2])

    @classmethod
    def from_tlwh(cls, tlwh: list) -> "Polygon":
        # convert to a bounding box, then to polygon
        top_x, top_y, width, height = tlwh
        bottom_x, bottom_y = top_x + width, top_y + height
        return Polygon.from_bbox([top_x, top_y, bottom_x, bottom_y])

    def to_proto(self) -> PolygonPb:
        """
        Generates a protobuf from the polygon

        Returns:
            PolygonPb: the generated polygon protobuf
        """
        polygon = PolygonPb()
        for vertex in self.vertices:
            v_proto = polygon.vertices.add()
            v_proto.x = vertex.x
            v_proto.y = vertex.y
        return polygon

    @classmethod
    def from_proto(cls, raw_proto: PolygonPb) -> "Polygon":
        """
        Creates a polygon from the corresponding protobuf

        Args:
            raw_proto (PolygonPb): proto to convert

        Returns:
            Polygon: the polygon to capture
        """
        vertices = [
            Point(x=vertex.x, y=vertex.y) for vertex in raw_proto.vertices
        ]
        return Polygon(vertices)

    def area(self) -> float:
        """
        Returns the bounding box area

        Returns:
            float: the bounding box area h * w
        """
        return self.to_shapely_polygon().area


@attr.s(slots=True)
class Pose:

    nose = attr.ib(default=None, type=KeyPoint)
    neck = attr.ib(default=None, type=KeyPoint)
    right_shoulder = attr.ib(default=None, type=KeyPoint)
    right_elbow = attr.ib(default=None, type=KeyPoint)
    right_wrist = attr.ib(default=None, type=KeyPoint)
    left_shoulder = attr.ib(default=None, type=KeyPoint)
    left_elbow = attr.ib(default=None, type=KeyPoint)
    left_wrist = attr.ib(default=None, type=KeyPoint)
    mid_hip = attr.ib(default=None, type=KeyPoint)
    right_hip = attr.ib(default=None, type=KeyPoint)
    right_knee = attr.ib(default=None, type=KeyPoint)
    right_ankle = attr.ib(default=None, type=KeyPoint)
    left_hip = attr.ib(default=None, type=KeyPoint)
    left_knee = attr.ib(default=None, type=KeyPoint)
    left_ankle = attr.ib(default=None, type=KeyPoint)
    right_eye = attr.ib(default=None, type=KeyPoint)
    left_eye = attr.ib(default=None, type=KeyPoint)
    right_ear = attr.ib(default=None, type=KeyPoint)
    left_ear = attr.ib(default=None, type=KeyPoint)
    left_big_toe = attr.ib(default=None, type=KeyPoint)
    left_small_toe = attr.ib(default=None, type=KeyPoint)
    left_heel = attr.ib(default=None, type=KeyPoint)
    right_big_toe = attr.ib(default=None, type=KeyPoint)
    right_small_toe = attr.ib(default=None, type=KeyPoint)
    right_heel = attr.ib(default=None, type=KeyPoint)

    def to_dict(self):
        """Convert pose into a dictionary"""
        return attr.asdict(self)

    @classmethod
    def from_inference_results(
        cls, keypoints: np.ndarray, confidences: np.ndarray
    ) -> "Pose":
        """Converts keypoints and confidences to Pose

        Args:
            keypoints (np.ndarray): keypoints (Nx17x2)
            confidences (np.ndarray): confidences (Nx17)
        Returns:
            Pose: pose
        """
        pose = cls()
        pose.nose = KeyPoint(
            x=keypoints[0][0], y=keypoints[0][1], confidence=confidences[0][0]
        )
        pose.left_eye = KeyPoint(
            x=keypoints[1][0], y=keypoints[1][1], confidence=confidences[1][0]
        )
        pose.right_eye = KeyPoint(
            x=keypoints[2][0], y=keypoints[2][1], confidence=confidences[2][0]
        )
        pose.left_ear = KeyPoint(
            x=keypoints[3][0], y=keypoints[3][1], confidence=confidences[3][0]
        )
        pose.right_ear = KeyPoint(
            x=keypoints[4][0], y=keypoints[4][1], confidence=confidences[4][0]
        )
        pose.left_shoulder = KeyPoint(
            x=keypoints[5][0], y=keypoints[5][1], confidence=confidences[5][0]
        )
        pose.right_shoulder = KeyPoint(
            x=keypoints[6][0], y=keypoints[6][1], confidence=confidences[6][0]
        )
        pose.left_elbow = KeyPoint(
            x=keypoints[7][0], y=keypoints[7][1], confidence=confidences[7][0]
        )
        pose.right_elbow = KeyPoint(
            x=keypoints[8][0], y=keypoints[8][1], confidence=confidences[8][0]
        )
        pose.left_wrist = KeyPoint(
            x=keypoints[9][0], y=keypoints[9][1], confidence=confidences[9][0]
        )
        pose.right_wrist = KeyPoint(
            x=keypoints[10][0],
            y=keypoints[10][1],
            confidence=confidences[10][0],
        )
        pose.left_hip = KeyPoint(
            x=keypoints[11][0],
            y=keypoints[11][1],
            confidence=confidences[11][0],
        )
        pose.right_hip = KeyPoint(
            x=keypoints[12][0],
            y=keypoints[12][1],
            confidence=confidences[12][0],
        )
        pose.left_knee = KeyPoint(
            x=keypoints[13][0],
            y=keypoints[13][1],
            confidence=confidences[13][0],
        )
        pose.right_knee = KeyPoint(
            x=keypoints[14][0],
            y=keypoints[14][1],
            confidence=confidences[14][0],
        )
        pose.left_ankle = KeyPoint(
            x=keypoints[15][0],
            y=keypoints[15][1],
            confidence=confidences[15][0],
        )
        pose.right_ankle = KeyPoint(
            x=keypoints[16][0],
            y=keypoints[16][1],
            confidence=confidences[16][0],
        )
        return pose

    @classmethod
    def from_dict(cls, data) -> "Pose":
        """Create a pose from a dictionary
        Args:
            data (dict): The dictionary to create the pose from
        Returns:
            Pose: The pose created from the dictionary
        """
        return Pose(
            nose=KeyPoint.from_dict(data["nose"]) if "nose" in data else None,
            neck=KeyPoint.from_dict(data["neck"]) if "neck" in data else None,
            right_shoulder=KeyPoint.from_dict(data["right_shoulder"])
            if "right_shoulder" in data
            else None,
            right_elbow=KeyPoint.from_dict(data["right_elbow"])
            if "right_elbow" in data
            else None,
            right_wrist=KeyPoint.from_dict(data["right_wrist"])
            if "right_wrist" in data
            else None,
            left_shoulder=KeyPoint.from_dict(data["left_shoulder"])
            if "left_shoulder" in data
            else None,
            left_elbow=KeyPoint.from_dict(data["left_elbow"])
            if "left_elbow" in data
            else None,
            left_wrist=KeyPoint.from_dict(data["left_wrist"])
            if "left_wrist" in data
            else None,
            mid_hip=KeyPoint.from_dict(data["mid_hip"])
            if "mid_hip" in data
            else None,
            right_hip=KeyPoint.from_dict(data["right_hip"])
            if "right_hip" in data
            else None,
            right_knee=KeyPoint.from_dict(data["right_knee"])
            if "right_knee" in data
            else None,
            right_ankle=KeyPoint.from_dict(data["right_ankle"])
            if "right_ankle" in data
            else None,
            left_hip=KeyPoint.from_dict(data["left_hip"])
            if "left_hip" in data
            else None,
            left_knee=KeyPoint.from_dict(data["left_knee"])
            if "left_knee" in data
            else None,
            left_ankle=KeyPoint.from_dict(data["left_ankle"])
            if "left_ankle" in data
            else None,
            right_eye=KeyPoint.from_dict(data["right_eye"])
            if "right_eye" in data
            else None,
            left_eye=KeyPoint.from_dict(data["left_eye"])
            if "left_eye" in data
            else None,
            right_ear=KeyPoint.from_dict(data["right_ear"])
            if "right_ear" in data
            else None,
            left_ear=KeyPoint.from_dict(data["left_ear"])
            if "left_ear" in data
            else None,
            left_big_toe=KeyPoint.from_dict(data["left_big_toe"])
            if "left_big_toe" in data
            else None,
            left_small_toe=KeyPoint.from_dict(data["left_small_toe"])
            if "left_small_toe" in data
            else None,
            left_heel=KeyPoint.from_dict(data["left_heel"])
            if "left_heel" in data
            else None,
            right_big_toe=KeyPoint.from_dict(data["right_big_toe"])
            if "right_big_toe" in data
            else None,
            right_small_toe=KeyPoint.from_dict(data["right_small_toe"])
            if "right_small_toe" in data
            else None,
            right_heel=KeyPoint.from_dict(data["right_heel"])
            if "right_heel" in data
            else None,
        )

    def to_proto(self) -> PosePb:
        """
        Generates the protobuf of the pose pb
        entries are not added if they are None

        Returns:
            PosePb: the generated protobuf
        """
        kwargs = {}

        def safe_update(name: str, value: KeyPointPb):
            """
            Updates the keyword arguments that get passed to the proto
            constructor

            Args:
                name (str): the name of the argument
                value (KeyPointPb): the value of the argument
            """

            if value is not None:
                kwargs[name] = value

        safe_update(
            "nose_keypoint", self.nose.to_proto() if self.nose else None
        )
        safe_update(
            "neck_keypoint", self.neck.to_proto() if self.neck else None
        )
        safe_update(
            "right_shoulder_keypoint",
            self.right_shoulder.to_proto() if self.right_shoulder else None,
        )
        safe_update(
            "right_elbow_keypoint",
            self.right_elbow.to_proto() if self.right_elbow else None,
        )
        safe_update(
            "right_wrist_keypoint",
            self.right_wrist.to_proto() if self.right_wrist else None,
        )
        safe_update(
            "left_shoulder_keypoint",
            self.left_shoulder.to_proto() if self.left_shoulder else None,
        )
        safe_update(
            "left_elbow_keypoint",
            self.left_elbow.to_proto() if self.left_elbow else None,
        )
        safe_update(
            "left_wrist_keypoint",
            self.left_wrist.to_proto() if self.left_wrist else None,
        )
        safe_update(
            "mid_hip_keypoint",
            self.mid_hip.to_proto() if self.mid_hip else None,
        )
        safe_update(
            "right_hip_keypoint",
            self.right_hip.to_proto() if self.right_hip else None,
        )
        safe_update(
            "right_knee_keypoint",
            self.right_knee.to_proto() if self.right_knee else None,
        )
        safe_update(
            "right_ankle_keypoint",
            self.right_ankle.to_proto() if self.right_ankle else None,
        )
        safe_update(
            "left_hip_keypoint",
            self.left_hip.to_proto() if self.left_hip else None,
        )
        safe_update(
            "left_knee_keypoint",
            self.left_knee.to_proto() if self.left_knee else None,
        )
        safe_update(
            "left_ankle_keypoint",
            self.left_ankle.to_proto() if self.left_ankle else None,
        )
        safe_update(
            "right_eye_keypoint",
            self.right_eye.to_proto() if self.right_eye else None,
        )
        safe_update(
            "left_eye_keypoint",
            self.left_eye.to_proto() if self.left_eye else None,
        )
        safe_update(
            "right_ear_keypoint",
            self.right_ear.to_proto() if self.right_ear else None,
        )
        safe_update(
            "left_ear_keypoint",
            self.left_ear.to_proto() if self.left_ear else None,
        )
        safe_update(
            "left_big_toe_keypoint",
            self.left_big_toe.to_proto() if self.left_big_toe else None,
        )
        safe_update(
            "left_small_toe_keypoint",
            self.left_small_toe.to_proto() if self.left_small_toe else None,
        )
        safe_update(
            "left_heel_keypoint",
            self.left_heel.to_proto() if self.left_heel else None,
        )
        safe_update(
            "right_big_toe_keypoint",
            self.right_big_toe.to_proto() if self.right_big_toe else None,
        )
        safe_update(
            "right_small_toe_keypoint",
            self.right_small_toe.to_proto() if self.right_small_toe else None,
        )
        safe_update(
            "right_heel_keypoint",
            self.right_heel.to_proto() if self.right_heel else None,
        )
        return PosePb(**kwargs)

    @classmethod
    def from_proto(cls, raw_proto: PosePb) -> "Pose":
        """
        Generates a pose struct based on the input proto

        Args:
            raw_proto (PosePb): the generated protobuf

        Returns:
            Pose: the pose generated from the protobuf
        """
        proto = VoxelProto(raw_proto)
        return Pose(
            nose=KeyPoint.from_proto(proto.nose_keypoint)
            if proto.nose_keypoint
            else None,
            neck=KeyPoint.from_proto(proto.neck_keypoint)
            if proto.neck_keypoint
            else None,
            right_shoulder=KeyPoint.from_proto(proto.right_shoulder_keypoint)
            if proto.right_shoulder_keypoint
            else None,
            right_elbow=KeyPoint.from_proto(proto.right_elbow_keypoint)
            if proto.right_elbow_keypoint
            else None,
            right_wrist=KeyPoint.from_proto(proto.right_wrist_keypoint)
            if proto.right_wrist_keypoint
            else None,
            left_shoulder=KeyPoint.from_proto(proto.left_shoulder_keypoint)
            if proto.left_shoulder_keypoint
            else None,
            left_elbow=KeyPoint.from_proto(proto.left_elbow_keypoint)
            if proto.left_elbow_keypoint
            else None,
            left_wrist=KeyPoint.from_proto(proto.left_wrist_keypoint)
            if proto.left_wrist_keypoint
            else None,
            mid_hip=KeyPoint.from_proto(proto.mid_hip_keypoint)
            if proto.mid_hip_keypoint
            else None,
            right_hip=KeyPoint.from_proto(proto.right_hip_keypoint)
            if proto.right_hip_keypoint
            else None,
            right_knee=KeyPoint.from_proto(proto.right_knee_keypoint)
            if proto.right_knee_keypoint
            else None,
            right_ankle=KeyPoint.from_proto(proto.right_ankle_keypoint)
            if proto.right_ankle_keypoint
            else None,
            left_hip=KeyPoint.from_proto(proto.left_hip_keypoint)
            if proto.left_hip_keypoint
            else None,
            left_knee=KeyPoint.from_proto(proto.left_knee_keypoint)
            if proto.left_knee_keypoint
            else None,
            left_ankle=KeyPoint.from_proto(proto.left_ankle_keypoint)
            if proto.left_ankle_keypoint
            else None,
            right_eye=KeyPoint.from_proto(proto.right_eye_keypoint)
            if proto.right_eye_keypoint
            else None,
            left_eye=KeyPoint.from_proto(proto.left_eye_keypoint)
            if proto.left_eye_keypoint
            else None,
            right_ear=KeyPoint.from_proto(proto.right_ear_keypoint)
            if proto.right_ear_keypoint
            else None,
            left_ear=KeyPoint.from_proto(proto.left_ear_keypoint)
            if proto.left_ear_keypoint
            else None,
            left_big_toe=KeyPoint.from_proto(proto.left_big_toe_keypoint)
            if proto.left_big_toe_keypoint
            else None,
            left_small_toe=KeyPoint.from_proto(proto.left_small_toe_keypoint)
            if proto.left_small_toe_keypoint
            else None,
            left_heel=KeyPoint.from_proto(proto.left_heel_keypoint)
            if proto.left_heel_keypoint
            else None,
            right_big_toe=KeyPoint.from_proto(proto.right_big_toe_keypoint)
            if proto.right_big_toe_keypoint
            else None,
            right_small_toe=KeyPoint.from_proto(proto.right_small_toe_keypoint)
            if proto.right_small_toe_keypoint
            else None,
            right_heel=KeyPoint.from_proto(proto.right_heel_keypoint)
            if proto.right_heel_keypoint
            else None,
        )

    def to_annotation_protos(self) -> ImageAnnotations:
        """
        Generates the protos for foxglove

        Returns:
            ImageAnnotations: the image annotation
        """

        ordering = [
            "nose",
            "left_eye",
            "right_eye",
            "left_ear",
            "right_ear",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
            "neck",
        ]

        l_pair = [
            (0, 1),
            (0, 2),
            (1, 3),
            (2, 4),  # Head
            (5, 6),
            (5, 7),
            (7, 9),
            (6, 8),
            (8, 10),
            (17, 11),
            (17, 12),  # Body
            (11, 13),
            (12, 14),
            (13, 15),
            (14, 16),
        ]

        p_color = [
            Color(r=0 / 255, g=255 / 255, b=255 / 255, a=0.5),
            Color(r=0 / 255, g=191 / 255, b=255 / 255, a=0.5),
            Color(r=0 / 255, g=255 / 255, b=102 / 255, a=0.5),
            Color(r=0 / 255, g=77 / 255, b=255 / 255, a=0.5),
            Color(
                r=0 / 255, g=255 / 255, b=0 / 255, a=0.5
            ),  # Nose, LEye, REye, LEar, REar
            Color(r=77 / 255, g=255 / 255, b=255 / 255, a=0.5),
            Color(r=77 / 255, g=255 / 255, b=204 / 255, a=0.5),
            Color(r=77 / 255, g=204 / 255, b=255 / 255, a=0.5),
            Color(r=191 / 255, g=255 / 255, b=77 / 255, a=0.5),
            Color(r=77 / 255, g=191 / 255, b=255 / 255, a=0.5),
            Color(
                r=191 / 255, g=255 / 255, b=77 / 255, a=0.5
            ),  # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
            Color(r=204 / 255, g=77 / 255, b=255 / 255, a=0.5),
            Color(r=77 / 255, g=255 / 255, b=204 / 255, a=0.5),
            Color(r=191 / 255, g=77 / 255, b=255 / 255, a=0.5),
            Color(r=77 / 255, g=255 / 255, b=191 / 255, a=0.5),
            Color(r=127 / 255, g=77 / 255, b=255 / 255, a=0.5),
            Color(r=77 / 255, g=255 / 255, b=127 / 255, a=0.5),
            Color(r=0 / 255, g=255 / 255, b=255 / 255, a=0.5),
        ]  # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
        line_color = [
            Color(r=0 / 255, g=215 / 255, b=255 / 255, a=0.5),
            Color(r=0 / 255, g=255 / 255, b=204 / 255, a=0.5),
            Color(r=0 / 255, g=134 / 255, b=255 / 255, a=0.5),
            Color(r=0 / 255, g=255 / 255, b=50 / 255, a=0.5),
            Color(r=77 / 255, g=255 / 255, b=222 / 255, a=0.5),
            Color(r=77 / 255, g=196 / 255, b=255 / 255, a=0.5),
            Color(r=77 / 255, g=135 / 255, b=255 / 255, a=0.5),
            Color(r=191 / 255, g=255 / 255, b=77 / 255, a=0.5),
            Color(r=77 / 255, g=255 / 255, b=77 / 255, a=0.5),
            Color(r=77 / 255, g=222 / 255, b=255 / 255, a=0.5),
            Color(r=255 / 255, g=156 / 255, b=127 / 255, a=0.5),
            Color(r=0 / 255, g=127 / 255, b=255 / 255, a=0.5),
            Color(r=255 / 255, g=127 / 255, b=77 / 255, a=0.5),
            Color(r=0 / 255, g=77 / 255, b=255 / 255, a=0.5),
            Color(r=255 / 255, g=77 / 255, b=36 / 255, a=0.5),
        ]

        circles = []
        circle_thickness = 1
        for idx, kp_name in enumerate(ordering):
            if idx < 5:
                continue
            kp_value = getattr(self, kp_name)
            if kp_value is not None:
                circles.append(
                    CircleAnnotation(
                        position=Point2(x=int(kp_value.x), y=int(kp_value.y)),
                        diameter=3,
                        outline_color=p_color[idx],
                        thickness=circle_thickness,
                    )
                )
        lines = []
        limb_thickness = 2
        for idx, limbs in enumerate(l_pair):
            if idx < 5:
                continue
            point_1 = getattr(self, ordering[limbs[0]])
            point_2 = getattr(self, ordering[limbs[1]])
            if point_1 is not None and point_2 is not None:

                points = [
                    Point2(x=int(p.x), y=int(p.y)) for p in [point_1, point_2]
                ]
                color = line_color[idx]

                line = PointsAnnotation(
                    type=2,
                    points=points,
                    outline_color=color,
                    thickness=limb_thickness,
                )
                lines.append(line)
        return ImageAnnotations(points=lines, circles=circles)


# TODO : Move the following to utils as conversion functions so we can
# standardize on one.


@attr.s(slots=True)
class RectangleXYXY:

    top_left_vertice = attr.ib(type=Point)
    bottom_right_vertice = attr.ib(type=Point)

    def __init__(self, top_left_vertice: Point, bottom_right_vertice: Point):
        self.top_left_vertice = top_left_vertice
        self.bottom_right_vertice = bottom_right_vertice

    def to_list(self):
        return [
            self.top_left_vertice.x,
            self.top_left_vertice.y,
            self.bottom_right_vertice.x,
            self.bottom_right_vertice.y,
        ]

    def to_polygon(self):
        data = {}
        data["vertices"] = [
            {"x": self.top_left_vertice.x, "y": self.top_left_vertice.y},
            {"x": self.top_left_vertice.x, "y": self.bottom_right_vertice.y},
            {
                "x": self.bottom_right_vertice.x,
                "y": self.bottom_right_vertice.y,
            },
            {"x": self.bottom_right_vertice.x, "y": self.top_left_vertice.y},
        ]
        return Polygon.from_dict(data)

    @classmethod
    def from_polygon(self, polygon):
        X = [vertice.x for vertice in polygon.vertices]
        Y = [vertice.y for vertice in polygon.vertices]
        return RectangleXYXY(Point(min(X), min(Y)), Point(max(X), max(Y)))

    @classmethod
    def from_list(self, data):
        return RectangleXYXY(
            Point(int(data[0]), int(data[1])),
            Point(int(data[2]), int(data[3])),
        )


@attr.s(slots=True)
class RectangleXYWH:

    top_left_vertice = attr.ib(type=Point)
    w = attr.ib(type=int)
    h = attr.ib(type=int)

    def __init__(self, top_left_vertice: Point, w: int, h: int):
        self.top_left_vertice = top_left_vertice
        self.w = w
        self.h = h

    def to_list(self):
        return [
            self.top_left_vertice.x,
            self.top_left_vertice.y,
            self.w,
            self.h,
        ]

    def to_polygon(self):
        data = {}
        data["vertices"] = [
            {"x": self.top_left_vertice.x, "y": self.top_left_vertice.y},
            {
                "x": self.top_left_vertice.x + self.w,
                "y": self.top_left_vertice.y,
            },
            {
                "x": self.top_left_vertice.x + self.w,
                "y": self.top_left_vertice.y + self.h,
            },
            {
                "x": self.top_left_vertice.x,
                "y": self.top_left_vertice.y + self.h,
            },
        ]

        return Polygon.from_dict(data)

    @classmethod
    def from_polygon(self, polygon):
        X = [int(vertice.x) for vertice in polygon.vertices]
        Y = [int(vertice.y) for vertice in polygon.vertices]
        return RectangleXYWH(
            Point(min(X), min(Y)), max(X) - min(X), max(Y) - min(Y)
        )

    @classmethod
    def from_list(self, data):
        return RectangleXYWH(
            Point(int(data[0]), int(data[1])), int(data[2]), int(data[3])
        )


@attr.s(slots=True)
class RectangleXCYCWH:

    center_vertice = attr.ib(type=Point)
    w = attr.ib(type=int)
    h = attr.ib(type=int)

    def __init__(self, center_vertice: Point, w: int, h: int):
        self.center_vertice = center_vertice
        self.w = w
        self.h = h

    def to_list(self):
        return [self.center_vertice.x, self.center_vertice.y, self.w, self.h]

    def to_polygon(self):
        data = {}
        data["vertices"] = [
            {
                "x": self.center_vertice.x - self.w / 2,
                "y": self.center_vertice.y - self.h / 2,
            },
            {
                "x": self.center_vertice.x - self.w / 2,
                "y": self.center_vertice.y + self.h / 2,
            },
            {
                "x": self.center_vertice.x + self.w / 2,
                "y": self.center_vertice.y - self.h / 2,
            },
            {
                "x": self.center_vertice.x + self.w / 2,
                "y": self.center_vertice.y + self.h / 2,
            },
        ]
        return Polygon.from_dict(data)

    @classmethod
    def from_polygon(self, polygon):
        X = [vertice.x for vertice in polygon.vertices]
        Y = [vertice.y for vertice in polygon.vertices]
        x, y, w, h = min(X), min(Y), max(X) - min(X), max(Y) - min(Y)
        return RectangleXCYCWH(Point(x + w / 2, y + h / 2), w, h)

    @classmethod
    def from_list(self, data):
        return RectangleXCYCWH(
            Point(int(data[0]), int(data[1])), int(data[2]), int(data[3])
        )
