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
import numpy as np
import torch
import torch.nn as nn

from core.perception.pose.DUC import DUC
from core.perception.pose.pPose_nms import pose_nms
from core.perception.pose.SE_Resnet import SEResnet
from core.structs.actor import ActorCategory
from core.structs.attributes import KeyPoint, Pose, RectangleXYXY
from lib.ml.inference.factory.base import InferenceBackendType
from lib.ml.inference.tasks.human_keypoint_detection_2d.alphapose.factory import (
    AlphaposeInferenceProviderFactory,
)

# trunk-ignore-begin(pylint/E0611)
# trunk can't see the generated protos
from protos.perception.graph_config.v1.perception_params_pb2 import (
    GpuRuntimeBackend,
)

# trunk-ignore-end(pylint/E0611)

"""
from core.labeling.loader import Loader
from core.cv.detector.api import YOLODetector
import cv2

detector = YOLODetector()
vcap = cv2.VideoCapture(Loader().get_video_url('8d10d83c-0b68-4e37-b31b-5e716ec94c30'))
ret, frame = vcap.read()
boxes, scores = detector(frame)
from core.cv.pose.api import Pose
pose = Pose()
pose(frame, boxes, scores)
"""


def _center_scale_to_box(center, scale):
    pixel_std = 1.0
    w = scale[0] * pixel_std
    h = scale[1] * pixel_std
    xmin = center[0] - w * 0.5
    ymin = center[1] - h * 0.5
    xmax = xmin + w
    ymax = ymin + h
    bbox = [xmin, ymin, xmax, ymax]
    return bbox


def heatmap_to_coord_simple(hms, bbox):
    coords, maxvals = get_max_pred(hms)

    hm_h = hms.shape[1]
    hm_w = hms.shape[2]

    # post-processing
    for p in range(coords.shape[0]):
        hm = hms[p]
        px = int(round(float(coords[p][0])))
        py = int(round(float(coords[p][1])))
        if 1 < px < hm_w - 1 and 1 < py < hm_h - 1:
            diff = np.array(
                (
                    hm[py][px + 1] - hm[py][px - 1],
                    hm[py + 1][px] - hm[py - 1][px],
                )
            )
            coords[p] += np.sign(diff) * 0.25

    preds = np.zeros_like(coords)

    # transform bbox to scale
    xmin, ymin, xmax, ymax = bbox
    w = xmax - xmin
    h = ymax - ymin
    center = np.array([xmin + w * 0.5, ymin + h * 0.5])
    scale = np.array([w, h])
    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = transform_preds(coords[i], center, scale, [hm_w, hm_h])

    return preds, maxvals


def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    target_coords[0:2] = affine_transform(coords[0:2], trans)
    return target_coords


def get_max_pred(heatmaps):
    num_joints = heatmaps.shape[0]
    width = heatmaps.shape[2]
    heatmaps_reshaped = heatmaps.reshape((num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 1)
    maxvals = np.max(heatmaps_reshaped, 1)

    maxvals = maxvals.reshape((num_joints, 1))
    idx = idx.reshape((num_joints, 1))

    preds = np.tile(idx, (1, 2)).astype(np.float32)

    preds[:, 0] = (preds[:, 0]) % width
    preds[:, 1] = np.floor((preds[:, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


def get_max_pred_batch(batch_heatmaps):
    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.max(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


def get_affine_transform(
    center,
    scale,
    rot,
    output_size,
    shift=np.array([0, 0], dtype=np.float32),
    inv=0,
):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.0]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def to_torch(ndarray):
    # numpy.ndarray => torch.Tensor
    if type(ndarray).__module__ == "numpy":
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError(
            "Cannot convert {} to torch tensor".format(type(ndarray))
        )
    return ndarray


def im_to_torch(img):
    """Transform ndarray image to torch tensor.

    Parameters
    ----------
    img: numpy.ndarray
        An ndarray with shape: `(H, W, 3)`.

    Returns
    -------
    torch.Tensor
        A tensor with shape: `(3, H, W)`.

    """
    img = np.transpose(img, (2, 0, 1))  # C*H*W
    img = to_torch(img).float()
    if img.max() > 1:
        img /= 255
    return img


def get_dir(src_point, rot_rad):
    """Rotate the point by `rot_rad` degree."""
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def _box_to_center_scale(x, y, w, h, aspect_ratio=1.0, scale_mult=1.25):
    """Convert box coordinates to center and scale.
    adapted from https://github.com/Microsoft/human-pose-estimation.pytorch
    """
    pixel_std = 1
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5

    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array(
        [w * 1.0 / pixel_std, h * 1.0 / pixel_std], dtype=np.float32
    )
    if center[0] != -1:
        scale = scale * scale_mult
    return center, scale


def get_3rd_point(a, b):
    """Return vector c that perpendicular to (a - b)."""
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


class PoseModel:
    def __init__(
        self,
        gpu_runtime: GpuRuntimeBackend,
        triton_server_url: str,
        model_path="artifacts_03_21_2023_pose_0630_jit_update/fast_res50_256x192.pt",
    ):
        self.inference_provider = AlphaposeInferenceProviderFactory(
            local_inference_provider_type=InferenceBackendType.TORCHSCRIPT,
            gpu_runtime=gpu_runtime,
            triton_server_url=triton_server_url,
        ).get_inference_provider(model_path)
        self.input_size = [256, 192]

    def get_affine_transform(
        self,
        center,
        scale,
        rot,
        output_size,
        shift=np.array([0, 0], dtype=np.float32),
        inv=0,
    ):
        if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
            scale = np.array([scale, scale])

        scale_tmp = scale
        src_w = scale_tmp[0]
        dst_w = output_size[0]
        dst_h = output_size[1]

        rot_rad = np.pi * rot / 180
        src_dir = get_dir([0, src_w * -0.5], rot_rad)
        dst_dir = np.array([0, dst_w * -0.5], np.float32)

        src = np.zeros((3, 2), dtype=np.float32)
        dst = np.zeros((3, 2), dtype=np.float32)
        src[0, :] = center + scale_tmp * shift
        src[1, :] = center + src_dir + scale_tmp * shift
        dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
        dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

        src[2:, :] = get_3rd_point(src[0, :], src[1, :])
        dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

        if inv:
            trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
        else:
            trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

        return trans

    def test_transform(self, src, bbox):
        xmin, ymin, xmax, ymax = bbox
        aspect_ratio = float(self.input_size[1]) / self.input_size[0]
        center, scale = _box_to_center_scale(
            xmin, ymin, xmax - xmin, ymax - ymin, aspect_ratio
        )
        scale = scale * 1.0

        input_size = self.input_size
        inp_h, inp_w = input_size

        trans = self.get_affine_transform(center, scale, 0, [inp_w, inp_h])
        img = cv2.warpAffine(
            src, trans, (int(inp_w), int(inp_h)), flags=cv2.INTER_LINEAR
        )
        bbox = _center_scale_to_box(center, scale)

        img = im_to_torch(img)
        img[0].add_(-0.406)
        img[1].add_(-0.457)
        img[2].add_(-0.480)

        return img, bbox

    def _process(self, frame, boxes, scores):
        with torch.no_grad():
            if len(boxes) == 0:
                return []

            inps = torch.zeros(len(boxes), 3, *self.input_size)
            cropped_boxes = torch.zeros(len(boxes), 4)

            for i, box in enumerate(boxes):
                inps[i], cropped_box = self.test_transform(frame, box)
                cropped_boxes[i] = torch.FloatTensor(cropped_box)

            hm_data = self.inference_provider.process(inps)
            # location prediction (n, kp, 2) | score prediction (n, kp, 1)
            pred = hm_data.cpu().data.numpy()
            assert pred.ndim == 4
            EVAL_JOINTS = [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
            ]

            pose_coords = []
            pose_scores = []
            for i in range(hm_data.shape[0]):
                bbox = cropped_boxes[i].tolist()
                pose_coord, pose_score = heatmap_to_coord_simple(
                    pred[i][EVAL_JOINTS], bbox
                )
                pose_coords.append(torch.from_numpy(pose_coord).unsqueeze(0))
                pose_scores.append(torch.from_numpy(pose_score).unsqueeze(0))
            preds_img = torch.cat(pose_coords)
            preds_scores = torch.cat(pose_scores)
            result = pose_nms(
                boxes,
                scores,
                torch.arange(len(scores)),
                preds_img,
                preds_scores,
                0,
            )
            return result

    def __call__(self, frame, frame_struct):
        """
        Plugin Pose and Detector Module
        Should we add option to load the GT as well when video is used?
        Determine the strcuts for Pose and BBox, should come from `core.structs`.
        Possibly a module in core.detector?
        """
        persons = []
        others = []

        for actor in frame_struct.actors:
            if actor.category == ActorCategory.PERSON:
                persons.append(actor)
            else:
                others.append(actor)

        frame_struct.actors = others

        if len(persons) == 0:
            return frame_struct

        boxes = []
        scores = []
        for person in persons:
            boxes.append(RectangleXYXY.from_polygon(person.polygon).to_list())
            scores.append(actor.confidence)

        boxes = torch.from_numpy(np.asarray(boxes))
        scores = torch.from_numpy(np.asarray(scores))
        results = (
            self._process(frame, boxes, scores) if boxes is not None else []
        )

        for result in results:
            person = persons[result["box_idx"]]
            person.pose = self._get_pose_from_result(result)
            frame_struct.actors.append(person)

        return frame_struct

    def _get_pose_from_result(self, result):
        keypoints = result.get("keypoints")
        kp_score = result.get("kp_score")
        if keypoints is not None and kp_score is not None:
            pose = Pose()
            pose.nose = KeyPoint(
                x=keypoints[0][0], y=keypoints[0][1], confidence=kp_score[0][0]
            )
            pose.left_eye = KeyPoint(
                x=keypoints[1][0], y=keypoints[1][1], confidence=kp_score[1][0]
            )
            pose.right_eye = KeyPoint(
                x=keypoints[2][0], y=keypoints[2][1], confidence=kp_score[2][0]
            )
            pose.left_ear = KeyPoint(
                x=keypoints[3][0], y=keypoints[3][1], confidence=kp_score[3][0]
            )
            pose.right_ear = KeyPoint(
                x=keypoints[4][0], y=keypoints[4][1], confidence=kp_score[4][0]
            )
            pose.left_shoulder = KeyPoint(
                x=keypoints[5][0], y=keypoints[5][1], confidence=kp_score[5][0]
            )
            pose.right_shoulder = KeyPoint(
                x=keypoints[6][0], y=keypoints[6][1], confidence=kp_score[6][0]
            )
            pose.left_elbow = KeyPoint(
                x=keypoints[7][0], y=keypoints[7][1], confidence=kp_score[7][0]
            )
            pose.right_elbow = KeyPoint(
                x=keypoints[8][0], y=keypoints[8][1], confidence=kp_score[8][0]
            )
            pose.left_wrist = KeyPoint(
                x=keypoints[9][0], y=keypoints[9][1], confidence=kp_score[9][0]
            )
            pose.right_wrist = KeyPoint(
                x=keypoints[10][0],
                y=keypoints[10][1],
                confidence=kp_score[10][0],
            )
            pose.left_hip = KeyPoint(
                x=keypoints[11][0],
                y=keypoints[11][1],
                confidence=kp_score[11][0],
            )
            pose.right_hip = KeyPoint(
                x=keypoints[12][0],
                y=keypoints[12][1],
                confidence=kp_score[12][0],
            )
            pose.left_knee = KeyPoint(
                x=keypoints[13][0],
                y=keypoints[13][1],
                confidence=kp_score[13][0],
            )
            pose.right_knee = KeyPoint(
                x=keypoints[14][0],
                y=keypoints[14][1],
                confidence=kp_score[14][0],
            )
            pose.left_ankle = KeyPoint(
                x=keypoints[15][0],
                y=keypoints[15][1],
                confidence=kp_score[15][0],
            )
            pose.right_ankle = KeyPoint(
                x=keypoints[16][0],
                y=keypoints[16][1],
                confidence=kp_score[16][0],
            )
            return pose
        return None


class FastPose(nn.Module):
    conv_dim = 128

    def __init__(
        self, norm_layer=nn.BatchNorm2d, num_layers=50, num_joints=17
    ):
        super(FastPose, self).__init__()
        self.preact = SEResnet(f"resnet{num_layers}")

        x = eval(f"tm.resnet{num_layers}(pretrained=True)")

        model_state = self.preact.state_dict()
        state = {
            k: v
            for k, v in x.state_dict().items()
            if k in self.preact.state_dict()
            and v.size() == self.preact.state_dict()[k].size()
        }
        model_state.update(state)
        self.preact.load_state_dict(model_state)

        self.suffle1 = nn.PixelShuffle(2)
        self.duc1 = DUC(512, 1024, upscale_factor=2, norm_layer=norm_layer)
        self.duc2 = DUC(256, 512, upscale_factor=2, norm_layer=norm_layer)

        self.conv_out = nn.Conv2d(
            self.conv_dim, num_joints, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        out = self.preact(x)
        out = self.suffle1(out)
        out = self.duc1(out)
        out = self.duc2(out)

        out = self.conv_out(out)
        return out

    def _initialize(self):
        for m in self.conv_out.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                # logger.info('=> init {}.bias as 0'.format(name))
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)
