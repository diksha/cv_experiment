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

import os
import wandb

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog

from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.data.datasets import register_coco_instances

from detectron2.checkpoint import DetectionCheckpointer

class CocoTrainer(DefaultTrainer):

  @classmethod
  def build_evaluator(cls, cfg, dataset_name, output_folder=None):

    if output_folder is None:
        os.makedirs("coco_eval", exist_ok=True)
        output_folder = "coco_eval"

    return COCOEvaluator(dataset_name, tasks=('bbox',), distributed=False, output_dir=output_folder, use_fast_impl=False,)



if __name__ == '__main__':
    wandb.init(project='detectron2_032022', name="eval", sync_tensorboard=True)

    register_coco_instances("voxel", {}, "/data/detectron2/cropped_coco/cropped_2/coco_labels.json", "/data/detectron2/cropped_coco/cropped_2")
    register_coco_instances("voxel_train", {}, "/data/detectron2/cropped_coco/cropped_2/train_labels.json", "/data/detectron2/cropped_coco/cropped_2")
    register_coco_instances("voxel_test", {}, "/data/detectron2/cropped_coco/cropped_2/test_labels.json", "/data/detectron2/cropped_coco/cropped_2")
    classes = ["PERSON", "PIT", "HARDHAT", "SAFETY_VEST"]


    print(len(DatasetCatalog.get("voxel")) )
    TOTAL_NUM_IMAGES = len(DatasetCatalog.get("voxel"))

    # training
    epochs = 50

    cfg = get_cfg()
    cfg.OUTPUT_DIR ="/home/vai_voxelsafety_com/models/detectron/1"
    # Get the basic model configuration from the model zoo 
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")) 
    # Passing the Train and Validation sets
    cfg.DATASETS.TRAIN = ("voxel_train",)
    cfg.DATASETS.TEST = ("voxel_test",)
    # Number of data loading threads
    cfg.DATALOADER.NUM_WORKERS = 4
    # initialize with pre-trained weights
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  
    # Number of images per batch across all machines.
    cfg.SOLVER.IMS_PER_BATCH = 8
    cfg.SOLVER.BASE_LR = 0.0125  # pick a good LearningRate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 16
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(classes) + 1

    cfg.SOLVER.NUM_GPUS = 1
    single_iteration = cfg.SOLVER.NUM_GPUS * cfg.SOLVER.IMS_PER_BATCH
    batches_per_epoch = int(TOTAL_NUM_IMAGES / single_iteration)
    cfg.SOLVER.MAX_ITER = 500  # Previously: batches_per_epoch * epochs

    print(f"output dir: {cfg.OUTPUT_DIR}")

    cfg.TEST.EVAL_PERIOD = batches_per_epoch # do eval once per epoch. 
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = CocoTrainer(cfg)
    trainer.resume_or_load(resume=False)
    checkpointer = DetectionCheckpointer(trainer.model, save_dir=cfg.OUTPUT_DIR)
    checkpointer.save("mymodel_before") 
    trainer.train()

    # Save the model
    checkpointer = DetectionCheckpointer(trainer.model, save_dir=cfg.OUTPUT_DIR)
    checkpointer.save("mymodel_after") 