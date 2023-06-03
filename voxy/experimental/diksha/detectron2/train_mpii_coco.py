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
import logging
import os
from collections import OrderedDict
import yaml
import json
from functools import partial
import wandb
import argparse

import detectron2.utils.comm as comm
from detectron2.config import get_cfg
from detectron2.data.datasets import load_coco_json
from detectron2.engine import launch, DefaultTrainer
from detectron2.structures import BoxMode
from detectron2.data.datasets import register_coco_instances
from detectron2 import model_zoo


logger = logging.getLogger("detectron2")

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

    for video_path in all_dataset_dict[dataset_group]:
        new_dict = get_dict_for_one_camera(video_path)
        dataset_dict_group += new_dict
    return dataset_dict_group


def main(args):
    epochs = args.epochs
    lr = args.lr
    data_dir = args.data_dir
    use_local_weights = (args.use_local_weights == 1)
    wandb_name = args.wandb_name + "_" + str(lr)
    out_dir = args.out_dir

    print(f'****** input parameters: epochs: {epochs}, lr = {lr}, out_dir: {out_dir}, data_dir: {data_dir}, wandb_name: {wandb_name}, use_local_weights: {use_local_weights}')

    wandb.init(project='detectron2', name=wandb_name, sync_tensorboard=True)

    register_coco_instances("voxel_mpii_as_coco_train", {}, os.path.join(args.data_dir,"coco_annotations","train.json"), os.path.join(args.data_dir))
    register_coco_instances("voxel_mpii_as_coco_val", {}, os.path.join(args.data_dir,"coco_annotations","val.json"), os.path.join(args.data_dir))
 

    TOTAL_NUM_IMAGES = len(load_coco_json(os.path.join(args.data_dir,"coco_annotations","train.json"), os.path.join(args.data_dir), "voxel_mpii_as_coco_train"))
    print("Total number of Train samples", TOTAL_NUM_IMAGES)

    cfg = get_cfg()
    cfg.OUTPUT_DIR = out_dir
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")) 
    cfg.DATASETS.TRAIN = ("voxel_mpii_as_coco_train",)
    cfg.DATASETS.TEST = ("voxel_mpii_as_coco_val",)
    cfg.DATALOADER.NUM_WORKERS = 4

    # initialize with pre-trained weights
    if use_local_weights:
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'model_final.pth' )
    else:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  

    # Number of images per batch across all machines.
    cfg.SOLVER.IMS_PER_BATCH = 8

    cfg.SOLVER.NUM_GPUS = args.num_gpus
    single_iteration = cfg.SOLVER.NUM_GPUS * cfg.SOLVER.IMS_PER_BATCH
    batches_per_epoch = int(TOTAL_NUM_IMAGES / single_iteration)
    cfg.SOLVER.MAX_ITER = batches_per_epoch * epochs

    cfg.LR_SCHEDULER_NAME = 'WarmupCosineLR'
    cfg.SOLVER.WARMUP_ITERS = int(cfg.SOLVER.MAX_ITER * 0.05)
    cfg.WARMUP_FACTOR = 1.0/1000
    cfg.SOLVER.BASE_LR = lr  
    cfg.SOLVER.STEPS = (int(cfg.SOLVER.MAX_ITER * 0.6), int(cfg.SOLVER.MAX_ITER * 0.8)) # reduce LR towards the end

    # cfg.SOLVER.MAX_ITER = 500  #No. of iterations   
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256  

    cfg.TEST.EVAL_PERIOD = batches_per_epoch # do eval once per epoch. 
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 # Set threshold for this model

    trainer = DefaultTrainer(cfg)

    trainer.resume_or_load(resume=False)
    trainer.train()


  
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='input args')
    parser.add_argument(
        "--portal_host", type=str, help="Hostname of the portal backend."
    )
    parser.add_argument("-epochs", help="number of epochs", type=int, default=20)
    parser.add_argument("-lr", help="learning rate", type=float, default=0.003)
    parser.add_argument("-out_dir", help="output folder", type=str, default='/home/diksha_voxelsafety_com/voxel/experimental/diksha/detectron2/output')
    parser.add_argument("-data_dir", help="data folder", type=str, default='/home/diksha_voxelsafety_com/mpii/')
    parser.add_argument("-use_local_weights", help="should use local weights or pre-trained COCO weights for training", type=int, default=0)
    parser.add_argument("-wandb_name", help="prj name", type=str, default='diksha_detectron_mpii_test')
    parser.add_argument("-num_gpus", help="number of GPUs", type=int, default=1)

    args = parser.parse_args()

    launch(
        main,
        args.num_gpus,
        dist_url="auto",
        args=(args,),
    )