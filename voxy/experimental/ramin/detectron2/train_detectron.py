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

# from detectron2.utils.logger import setup_logger
# setup_logger()

import numpy as np
import cv2
import random
import yaml
import os
import json
from functools import partial
import wandb

import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator


class CocoTrainer(DefaultTrainer):

  @classmethod
  def build_evaluator(cls, cfg, dataset_name, output_folder=None):

    if output_folder is None:
        os.makedirs("coco_eval", exist_ok=True)
        output_folder = "coco_eval"

    return COCOEvaluator(dataset_name, tasks=('bbox',), distributed=False, output_dir=output_folder)


def get_dict_for_one_camera(video_path):
    full_data_dir = os.path.expanduser(video_path)

    json_file = os.path.join(full_data_dir, "coco_labels.json")
    with open(json_file) as f:
        dataset_dicts = json.load(f)
    for i in dataset_dicts:
        filename = i["file_name"].split('/')[-1] 
        i["file_name"] = os.path.join(full_data_dir, "images", filename) 
        for j in i["annotations"]:
            j["bbox_mode"] = BoxMode.XYWH_ABS 
            j["category_id"] = int(j["category_id"])
    return dataset_dicts

def get_dict_for_all_cameras(data_dir, dataset_group='train'):
    dataset_dict_group = []

    full_data_dir = os.path.expanduser(data_dir)
    yaml_path = os.path.join(full_data_dir, 'all_classes.yaml')

    with open(yaml_path) as f:
        all_dataset_dict = yaml.safe_load(f)   

#     print('all_dataset_dict', all_dataset_dict)
    for video_path in all_dataset_dict[dataset_group]:
        new_dict = get_dict_for_one_camera(video_path)
        dataset_dict_group += new_dict

    return dataset_dict_group




if __name__ == '__main__':
    wandb.init(project='detectron2', name="eval3", sync_tensorboard=True)

    dataset_dict_group = get_dict_for_all_cameras(data_dir='~/data', dataset_group='train')

    dataset_dict_group = get_dict_for_all_cameras(data_dir='~/data', dataset_group='train')


    train_val_func = partial(get_dict_for_all_cameras, data_dir='~/data')
    classes = ["PERSON", "PIT", "HARDHAT", "SAFETY_VEST"]

    # Registering the Dataset
    for d in ["train", "val"]:
        DatasetCatalog.register("voxel_" + d, lambda d=d: train_val_func(dataset_group=d))
        MetadataCatalog.get("voxel_" + d).set(thing_classes=classes)


    train_metadata = MetadataCatalog.get("voxel_train")
    val_metadata = MetadataCatalog.get("voxel_val")

    d = get_dict_for_all_cameras(data_dir='~/data', dataset_group='train');
    TOTAL_NUM_IMAGES = len(d)

    # training
    epochs = 50

    cfg = get_cfg()
    # Get the basic model configuration from the model zoo 
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")) 
    # Passing the Train and Validation sets
    cfg.DATASETS.TRAIN = ("voxel_train",)
    cfg.DATASETS.TEST = ("voxel_val",)
    # Number of data loading threads
    cfg.DATALOADER.NUM_WORKERS = 4
    # initialize with pre-trained weights
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  
    # Number of images per batch across all machines.
    cfg.SOLVER.IMS_PER_BATCH = 8
    cfg.SOLVER.BASE_LR = 0.0125  # pick a good LearningRate
    # cfg.SOLVER.MAX_ITER = 500  #No. of iterations   
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256  
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(classes)

    cfg.SOLVER.NUM_GPUS = 1
    single_iteration = cfg.SOLVER.NUM_GPUS * cfg.SOLVER.IMS_PER_BATCH
    batches_per_epoch = int(TOTAL_NUM_IMAGES / single_iteration)
    cfg.SOLVER.MAX_ITER = batches_per_epoch * epochs

    cfg.TEST.EVAL_PERIOD = batches_per_epoch # do eval once per epoch. 
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = CocoTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()