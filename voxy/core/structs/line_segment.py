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
"""Wrapper around Sympy module."""

# For python 3 compatibility
from __future__ import absolute_import, division, print_function

import math

import sympy.geometry as sympy

from core.utils.constants import FLOATING_POINT_ERROR_THRESHOLD
from core.utils.math import compare


class Point:
    """Represents a geometric point."""

    def __init__(self, x, y):
        self.point = sympy.Point(x, y)

    def __str__(self):
        return "X: {}, Y: {}".format(self.point.x, self.point.y)

    def to_tuple(self):
        return (self.point.x, self.point.y)

    def is_equal(self, p2):
        if compare.is_close(
            self.x, p2.x, FLOATING_POINT_ERROR_THRESHOLD
        ) and compare.is_close(self.y, p2.y, FLOATING_POINT_ERROR_THRESHOLD):
            return True
        return False

    @property
    def x(self):
        return self.point.x

    @property
    def y(self):
        return self.point.y


class LineSegment:
    """Represents a geometric line segment."""

    def __init__(self, point1, point2):
        self.segment = sympy.Segment(point1.point, point2.point)
        self.p1 = point1
        self.p2 = point2

    def __str__(self):
        return "Point1: {}, Point2: {}".format(
            self.p1.__str__(), self.p2.__str__()
        )

    def length(self):
        """Returns the length of the line segment."""
        if isinstance(self.segment, sympy.Segment):
            return self.segment.length
        else:
            return 0

    def slope(self):
        if isinstance(self.segment, sympy.Segment):
            return self.segment.slope
        else:
            # This is incorrect. Needs fixing
            return 0

    def get_linesegment_with_angle_displacement(self, pivot, angle):
        """Gets line segment with angle displacement.

        Returns:
            LineSegment: line segment that is displaced by `angle` around
                point `pivot` from the input line.
        """
        angle = math.radians(angle)
        temp_segment = self.segment.rotate(angle, pivot.point)
        temp_p1 = Point(temp_segment.p1.x, temp_segment.p1.y)
        temp_p2 = Point(temp_segment.p2.x, temp_segment.p2.y)
        return LineSegment(temp_p1, temp_p2)

    def get_parallel_line_segment_originating_at_point(self, point):
        """Gets parallel line segment originating at the specified point.

        We calculate the second point here as being strictly
        above (negative y direction) from "point". This is an
        important assumption that serves the current purpose but
        should be made explicit in the function call. The new point is
        calculated as
        x = x0 +/- lenght*(sqrt(1/(1+m^2)))
        y = y0 +/- m*lenght*(sqrt(1/(1+m^2)))
        where m is the slope the line

        Returns:
            LineSegment: line segment originating at the specified point.
        """
        # TODO(harishma): handle infinity case
        # line = self.segment.parallel_line(point.point)
        m = self.slope()
        length = self.length()
        coefficient = length * math.sqrt((1.0 / (1 + math.pow(m, 2))))
        x0 = point.point.x
        y0 = point.point.y
        x1 = x0 + coefficient
        x2 = x0 - coefficient
        y1 = y0 + m * coefficient
        y2 = y0 - m * coefficient
        new_point = None
        if y1 < y0:
            new_point = Point(x1, y1)
        else:
            new_point = Point(x2, y2)
        return LineSegment(new_point, point)

    def get_intersecting_point(self, l2):
        if not sympy.Line.are_concurrent(self.segment, l2.segment):
            return None  # any point will do
        if isinstance(
            sympy.intersection(self.segment, l2.segment)[0], sympy.Point
        ):
            temp = sympy.intersection(self.segment, l2.segment)[0]
            return Point(temp.x, temp.y)
        return None

    def get_non_intersecting_point(self, l2):
        if not sympy.Line.are_concurrent(self.segment, l2.segment):
            return self.p1  # any point will do
        point_of_intersection = self.get_intersecting_point(l2)
        if self.p1.is_equal(point_of_intersection):
            return self.p2
        return self.p1

    def get_endpoint_not_equal_to(self, p):
        if not self.p1.is_equal(p):
            return self.p1
        if not self.p2.is_equal(p):
            return self.p2
        return None

    def angle_between(self, l2):
        return self.segment.angle_between(l2.segment)

    def smallest_angle_between(self, l2):
        return self.segment.smallest_angle_between(l2.segment)

    def projection(self, p):
        return self.segment.projection(p.point)

    def get_point_along_y_axis_direction_at_distance(
        self, distance, starting_point
    ):
        if not isinstance(starting_point.point, sympy.Point):
            raise TypeError("Starting Point must of the type Point")

        new_point = Point(starting_point.x, starting_point.y + distance)
        p = self.projection(new_point)
        return Point(p.x, p.y)

    @staticmethod
    def transform_line_segements_with_x_displacement(
        segment_to_rotate, segment_to_displace, new_x_coordinate
    ):
        """Transforms line segment with X displacement.

        Returns:
            Tuple[LineSegment, LineSegment]: two new line segments, with the
                first line rotated about the non-intersecting point so as to
                be able to continue to intersect with the second and the
                second line translated along x axis to the new x coordinate.
        """
        if not sympy.Line.are_concurrent(
            segment_to_rotate.segment, segment_to_displace.segment
        ):
            # improper use of the function
            print(
                "linesegements_with_x_displacement sent non intersecting line segments."
            )
            return None

        non_intersecting_point = (
            segment_to_displace.get_non_intersecting_point(segment_to_rotate)
        )
        new_point = Point(new_x_coordinate, non_intersecting_point.y)
        displaced_segment = (
            segment_to_displace.get_parallel_line_segment_originating_at_point(
                new_point
            )
        )

        start_point = segment_to_rotate.get_endpoint_not_equal_to(
            segment_to_rotate.get_intersecting_point(segment_to_displace)
        )
        end_point = displaced_segment.get_endpoint_not_equal_to(new_point)

        rotated_segment = LineSegment(start_point, end_point)

        return (rotated_segment, displaced_segment)

    @staticmethod
    def transform_line_segments_with_angular_displacement(
        segment_to_rotate, segment_to_displace, angular_displacement
    ):
        """Transform line segment with angular displacement.

        Returns:
            Tuple[LineSegment, LineSegment]: two new line segments, forming
                the same angle as the input line segments and with the second
                line displaced along x axis to the new x coordinate.
        """
        if not sympy.Line.are_concurrent(
            segment_to_rotate.segment, segment_to_displace.segment
        ):
            # improper use of the function
            print(
                "linesegments_with_angular_displacement sent non intersecting line segments"
            )
            return None

        angle_between_segments = segment_to_rotate.angle_between(
            segment_to_displace
        )
        non_intersecting_point = segment_to_rotate.get_non_intersecting_point(
            segment_to_displace
        )
        rotated_segment = (
            segment_to_rotate.get_linesegment_with_angle_displacement(
                non_intersecting_point, angular_displacement
            )
        )
        start_point = rotated_segment.get_endpoint_not_equal_to(
            non_intersecting_point
        )
        displaced_segment = (
            segment_to_displace.get_parallel_line_segment_originating_at_point(
                start_point
            )
        )
        current_angle_between_segments = rotated_segment.angle_between(
            displaced_segment
        )
        angle_to_rotate = (
            angle_between_segments - current_angle_between_segments
        )
        displaced_segment = (
            displaced_segment.get_linesegment_with_angle_displacement(
                start_point, angle_to_rotate
            )
        )

        return (rotated_segment, displaced_segment)
