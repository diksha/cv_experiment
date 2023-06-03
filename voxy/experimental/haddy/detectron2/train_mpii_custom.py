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
import argparse
import os
import numpy as np
import cv2 as cv

from functools import partial

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer

import scipy.io as sio
import wandb

def get_frame_dict(data_dir, obj_list):
    frame = {}
    rect = obj_list.annorect
    img_d = obj_list.image
    obj_img = img_d[0, 0]
    name_d = obj_img.name
    image_dir = os.path.join(data_dir, "images")
    if os.path.isfile(os.path.join(image_dir, name_d[0])):
        frame["file_name"] = os.path.join(image_dir, name_d[0])
        frame["image_id"] = name_d[0]
        img = cv.imread(os.path.join(image_dir, name_d[0]))
        frame["height"] = img.shape[0]
        frame["width"] = img.shape[1]
        annotations = []
        if rect.shape[0] != 0:  # check if a bbox exists in img
            for ridx in range(0, rect.shape[1]):
                annotation = {}
                obj_rect = rect[0, ridx]
                if 'annopoints' not in obj_rect._fieldnames:
                    continue
                annopoints = obj_rect.annopoints
                x1_d = obj_rect.x1
                x1 = x1_d[0][0]
                x2_d = obj_rect.x2
                x2 = x2_d[0][0]
                y1_d = obj_rect.y1
                y1 = y1_d[0][0]
                y2_d = obj_rect.y2
                y2 = y2_d[0][0]
                annotation["bbox"] = [x1, y1, x2, y2]
                if annopoints.shape[0] == 0:
                    continue
                obj_points = annopoints[0, 0]
                points = obj_points.point
                keypoints = [0] * 48
                for px in range(0, points.shape[1]):
                    po = points[0, px]
                    po_id_d = po.id
                    po_ind = po_id_d[0][0] * 3                    
                    if 'is_visible' not in po._fieldnames:
                        keypoints[po_ind + 2] = 1
                    else:
                        po_v_d = po.is_visible
                        if po_v_d.shape[0] == 1:
                            keypoints[po_ind + 2] = int(po_v_d[0][0]) + 1
                        else:
                            keypoints[po_ind + 2] = 2
                    po_x_d = po.x
                    po_y_d = po.y
                    keypoints[po_ind] = po_x_d[0][0]
                    keypoints[po_ind + 1] = po_y_d[0][0]
                annotation["bbox_mode"] = 0  # BoxMode.XYXY_ABS
                annotation["category_id"] = 0
                annotation["keypoints"] = keypoints
                annotations.append(annotation)
        frame["annotations"] = annotations
    return frame


def get_data_dicts(data_dir, dataset_group="train"):
    dataset_dicts = []
    mat = sio.loadmat(os.path.join(data_dir, "mpii_human_pose_v1_u12_2", "mpii_human_pose_v1_u12_1"), struct_as_record=False)
    rel = mat['RELEASE']
    obj_rel = rel[0, 0]
    annolist = obj_rel.annolist
    img_train = obj_rel.img_train
    n = annolist.shape[1]

    for ix in range(0, n):
        if (dataset_group=="train" and img_train[0, ix] == 1) or (dataset_group=="val" and img_train[0, ix] == 0):  # check if assigned to train or test set
            obj_list = annolist[0, ix]
            dataset_dicts.append(get_frame_dict(data_dir, obj_list))
    return dataset_dicts


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='input args')
    parser.add_argument(
        "--portal_host", type=str, help="Hostname of the portal backend."
    )
    parser.add_argument("-epochs", help="number of epochs", type=int, default=20)
    parser.add_argument("-lr", help="learning rate", type=float, default=0.003)
    parser.add_argument("-out_dir", help="output folder", type=str, default='/home/haddy/voxel/experimental/haddy/detectron2/output')
    parser.add_argument("-data_dir", help="data folder", type=str, default='/home/haddy/mpii/')
    parser.add_argument("-use_local_weights", help="should use local weights or pre-trained COCO weights for training", type=int, default=0)
    parser.add_argument("-wandb_name", help="prj name", type=str, default='haddy_detectron_mpii_test')
    parser.add_argument("-num_gpus", help="number of GPUs", type=int, default=1)

    args = parser.parse_args()
    epochs = args.epochs
    lr = args.lr
    data_dir = args.data_dir
    use_local_weights = (args.use_local_weights == 1)
    wandb_name = args.wandb_name + "_" + str(lr)
    out_dir = args.out_dir

    wandb.init(project='detectron2', name=wandb_name, sync_tensorboard=True)

    get_data_dicts_func = partial(get_data_dicts, data_dir=data_dir)

    MPII_PERSON_KEYPOINT_NAMES = ["r ankle", "r knee", "r hip", "l hip", "l knee", "l ankle", "pelvis", "thorax", "upper neck", "head top", "r wrist", "r elbow", "r shoulder", "l shoulder", "l elbow", "l wrist"] 
  
    # Pairs of keypoints that should be exchanged under horizontal flipping 
    MPII_PERSON_KEYPOINT_FLIP_MAP = ( 
        ("r ankle", "l ankle"), 
        ("r knee", "l knee"), 
        ("r hip", "l hip"), 
        ("r wrist", "l wrist"), 
        ("r elbow", "l elbow"), 
        ("r shoulder", "l shoulder"), 
    ) 
    
    # Registering the Dataset
    for d in ["train", "val"]:
        DatasetCatalog.register("voxel_mpii_as_custom_" + d, lambda d=d: get_data_dicts_func(dataset_group=d))
        MetadataCatalog.get("voxel_mpii_as_custom_" + d).keypoint_names = MPII_PERSON_KEYPOINT_NAMES
        MetadataCatalog.get("voxel_mpii_as_custom_" + d).keypoint_flip_map = MPII_PERSON_KEYPOINT_FLIP_MAP

    d_train = get_data_dicts(data_dir=data_dir, dataset_group='train')
    d_val = get_data_dicts(data_dir=data_dir, dataset_group='val')
    print(f'training images = {len(d_train)}, val images = {len(d_val)}')
    TOTAL_NUM_IMAGES = len(d_train)

    cfg = get_cfg()
    cfg.OUTPUT_DIR = out_dir
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_1x.yaml"))
    cfg.DATASETS.TRAIN = ("voxel_mpii_as_custom_train",)
    cfg.DATASETS.TEST = ("voxel_mpii_as_custom_val",)
   
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_1x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 8

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 # only person class
    cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 16 

    cfg.SOLVER.NUM_GPUS = args.num_gpus
    single_iteration = cfg.SOLVER.NUM_GPUS * cfg.SOLVER.IMS_PER_BATCH
    batches_per_epoch = int(TOTAL_NUM_IMAGES / single_iteration)
    cfg.SOLVER.MAX_ITER = batches_per_epoch * epochs

    cfg.LR_SCHEDULER_NAME = 'WarmupCosineLR'
    cfg.SOLVER.WARMUP_ITERS = int(cfg.SOLVER.MAX_ITER * 0.05)
    cfg.WARMUP_FACTOR = 1.0/1000
    cfg.SOLVER.BASE_LR = lr
    cfg.SOLVER.STEPS = (int(cfg.SOLVER.MAX_ITER * 0.6), int(cfg.SOLVER.MAX_ITER * 0.8)) # reduce LR towards the end
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 # Set threshold for this model

    cfg.TEST.EVAL_PERIOD = batches_per_epoch # do eval once per epoch. 
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()