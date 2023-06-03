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
import cv2

color_palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def compute_color_for_labels(label):
    """Simple function that adds fixed color depending on the class."""
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in color_palette]
    return tuple(color)


def draw_bounding_boxes_with_ids(img, bbox, identities=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(coord) for coord in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(id)
        label = "{}{:d}".format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(
            img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1
        )
        cv2.putText(
            img,
            label,
            (x1, y1 + t_size[1] + 4),
            cv2.FONT_HERSHEY_PLAIN,
            2,
            [255, 255, 255],
            2,
        )
    return img


def draw_bounding_box_with_id(img, bbox, id=None, offset=(0, 0)):
    x1, y1, x2, y2 = [int(coord) for coord in bbox]
    x1 += offset[0]
    x2 += offset[0]
    y1 += offset[1]
    y2 += offset[1]
    # box text and bar
    id_int = 0
    if isinstance(id, list) and len(id) > 1:
        id_int = int(id[0]) if id is not None else 0
        label = "{}{:s}".format("", ",".join(str(id)))
    else:
        id_int = int(id) if id is not None else 0
        label = "{}{:d}".format("", id_int)
    color = compute_color_for_labels(id_int)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    if id is not None:
        cv2.rectangle(
            img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1
        )
        cv2.putText(
            img,
            label,
            (x1, y1 + t_size[1] + 4),
            cv2.FONT_HERSHEY_PLAIN,
            2,
            [255, 255, 255],
            2,
        )
    return img


def draw_bounding_box_with_tag(img, bbox, tag=None, offset=(0, 0)):
    x1, y1, x2, y2 = [int(coord) for coord in bbox]
    x1 += offset[0]
    x2 += offset[0]
    y1 += offset[1]
    y2 += offset[1]
    # box text and bar
    # color = compute_color_for_labels(0)
    color = (0, 0, 200)
    t_size = cv2.getTextSize(tag, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    if id is not None:
        cv2.rectangle(
            img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1
        )
        cv2.putText(
            img,
            tag,
            (x1, y1 + t_size[1] + 4),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            [255, 255, 255],
            2,
        )
    return img


def draw_bounding_boxes(img, bbox, offset=(0, 0), color_id=0):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(coord) for coord in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        color = compute_color_for_labels(color_id)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
    return img


def draw_bounding_box(img, bbox, offset=(0, 0), color_id=255):
    x1, y1, x2, y2 = [int(coord) for coord in bbox]
    x1 += offset[0]
    x2 += offset[0]
    y1 += offset[1]
    y2 += offset[1]
    color = compute_color_for_labels(color_id)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
    return img


def draw_bounding_boxes_xywh(img, bbox, offset=(0, 0), color_id=0):
    bbox_xyxy = []
    for i, box in enumerate(bbox):
        x1, y1, w, h = [int(coord) for coord in box]
        x2 = x1 + w
        y2 = y1 + h
        bbox_xyxy.append([x1, y1, x2, y2])
    return draw_bounding_boxes(img, bbox_xyxy, offset, color_id)


def draw_bounding_boxes_xcycwh(img, bbox, offset=(0, 0), color_id=0):
    im_height, im_width = img.shape[:2]
    bbox_xyxy = []
    for i, box in enumerate(bbox):
        x, y, w, h = [int(coord) for coord in box]
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), im_width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), im_height - 1)
        bbox_xyxy.append([x1, y1, x2, y2])
    return draw_bounding_boxes(img, bbox_xyxy, offset, color_id)
