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
import torch
from torch.nn.parallel import DistributedDataParallel
import yaml
import json
from functools import partial
import wandb
import argparse

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.engine import default_argument_parser, default_setup, default_writers, launch, DefaultTrainer
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import EventStorage
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances


logger = logging.getLogger("detectron2")


def build_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = 'coco' 
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    elif len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return build_evaluator(cfg, dataset_name, output_folder)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

def get_dict_for_one_video(video_path):
    # video_path = video_path.replace('/data', '~/data')   #DEBUG -- remove this

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

def get_dict_for_all_videos(data_dir, dataset_group='train'):
    dataset_dict_group = []

    full_data_dir = os.path.expanduser(data_dir)
    yaml_path = os.path.join(full_data_dir, 'all_classes.yaml')

    with open(yaml_path) as f:
        all_dataset_dict = yaml.safe_load(f)   

    for video_path in all_dataset_dict[dataset_group]:
        new_dict = get_dict_for_one_video(video_path)
        dataset_dict_group += new_dict

    # # synth_dir = '/home/ramin_voxelsafety_com/data/synth'
    # synth_dir = '/data/synth'
    # if dataset_group == 'train':
    #     json_file = os.path.join(synth_dir, "coco_labels_custom.json")
    #     with open(json_file) as f:
    #         dataset_dicts = json.load(f)

    #     for i in dataset_dicts:
    #         filename = i["file_name"].split('/')[-1] 
    #         i["file_name"] = os.path.join(synth_dir, "images", filename) 

    #     dataset_dict_group += dataset_dicts
    return dataset_dict_group


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='input args')
    parser.add_argument(
        "--portal_host", type=str, help="Hostname of the portal backend."
    )
    parser.add_argument("-epochs", help="number of epochs", type=int, default=20)
    parser.add_argument("-lr", help="learning rate", type=float, default=0.003)
    parser.add_argument("-out_dir", help="output folder", type=str, default='/home/ramin_voxelsafety_com/voxel/experimental/ramin/detectron2/output')
    parser.add_argument("-data_dir", help="data folder", type=str, default='~/data')
    parser.add_argument("-use_local_weights", help="should use local weights or pre-trained COCO weights for training", type=int, default=0)
    parser.add_argument("-wandb_name", help="prj name", type=str, default='my_detectron')
    parser.add_argument("-num_gpus", help="number of GPUs", type=int, default=1)
    parser.add_argument("-gamma", help="lr gamma", type=float, default=0.1)
    parser.add_argument("-batch_size", help="batch size", type=int, default=8)

    args = parser.parse_args()
    epochs = args.epochs
    lr = args.lr
    data_dir = args.data_dir
    use_local_weights = (args.use_local_weights == 1)
    wandb_name = args.wandb_name + "_" + str(lr)
    out_dir = args.out_dir
    gamma = args.gamma
    batch_size = args.batch_size

    print(f'****** input parameters: epochs: {epochs}, lr = {lr}, out_dir: {out_dir}, data_dir: {data_dir}, wandb_name: {wandb_name}, use_local_weights: {use_local_weights}, gamma: {gamma}, batch size = {batch_size}')

    wandb.init(project='detectron2', name=wandb_name, sync_tensorboard=True)

    d_train = get_dict_for_all_videos(data_dir=data_dir, dataset_group='train')
    d_val = get_dict_for_all_videos(data_dir=data_dir, dataset_group='val')
    print(f'training images = {len(d_train)}, val images = {len(d_val)}')
    TOTAL_NUM_IMAGES = len(d_train)

    classes = ["PERSON", "PIT", "HARDHAT", "SAFETY_VEST"]
    # classes = ["person", "pit", "hard_hat", "safety_vest"]
    train_val_func = partial(get_dict_for_all_videos, data_dir=data_dir)

    # Registering the Dataset
    for d in ["train", "val"]:
        DatasetCatalog.register("voxel_" + d, lambda d=d: train_val_func(dataset_group=d))
        MetadataCatalog.get("voxel_" + d).set(thing_classes=classes)

    # register_coco_instances("synthetic_train", {}, os.path.join(data_dir, 'synth', "coco_labels.json"), os.path.join(data_dir,'synth', "images"))
    # MetadataCatalog.get("synthetic_train").set(thing_classes=classes)

    register_coco_instances("coco_person", {}, os.path.join(data_dir, 'coco_person', "coco_labels_filtered.json"), os.path.join(data_dir,'coco_person', "images"))
    MetadataCatalog.get("coco_person").set(thing_classes=classes)

    train_metadata = MetadataCatalog.get("voxel_train")
    val_metadata = MetadataCatalog.get("voxel_val")

    cfg = get_cfg()
    cfg.OUTPUT_DIR = out_dir
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")) 
    # cfg.DATASETS.TRAIN = ("voxel_train", )
    cfg.DATASETS.TRAIN = ("voxel_train", "coco_person")
    # cfg.DATASETS.TRAIN = ("voxel_train", "synthetic_train")
    cfg.DATASETS.TEST = ("voxel_val",)
    cfg.DATALOADER.NUM_WORKERS = 4

    # initialize with pre-trained weights
    if use_local_weights:
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'model_final.pth' )
    else:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  

    # Number of images per batch across all machines.
    cfg.SOLVER.IMS_PER_BATCH = batch_size

    cfg.SOLVER.NUM_GPUS = args.num_gpus
    single_iteration = cfg.SOLVER.NUM_GPUS * cfg.SOLVER.IMS_PER_BATCH
    batches_per_epoch = int(TOTAL_NUM_IMAGES / single_iteration)
    cfg.SOLVER.MAX_ITER = batches_per_epoch * epochs

    cfg.LR_SCHEDULER_NAME = 'WarmupCosineLR'
    cfg.SOLVER.WARMUP_ITERS = int(cfg.SOLVER.MAX_ITER * 0.05)
    # cfg.WARMUP_FACTOR = 1.0/10
    cfg.SOLVER.BASE_LR = lr  
    cfg.SOLVER.STEPS = (int(cfg.SOLVER.MAX_ITER * 0.6), int(cfg.SOLVER.MAX_ITER * 0.8), int(cfg.SOLVER.MAX_ITER * 0.9), int(cfg.SOLVER.MAX_ITER * 0.95)) # reduce LR towards the end
    cfg.SOLVER.GAMMA = gamma

    # cfg.SOLVER.MAX_ITER = 500  #No. of iterations   
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256  
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(classes)

    cfg.TEST.EVAL_PERIOD = batches_per_epoch # do eval once per epoch. 
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 # Set threshold for this model

    # MODEL.ANCHOR_GENERATOR.SIZES = [[32], [64], [128], [256], [512]]
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[16, 32], [32, 64], [64, 128], [128, 256], [256, 512]]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]

    trainer = Trainer(cfg)

    trainer.resume_or_load(resume=False)
    trainer.train()


  