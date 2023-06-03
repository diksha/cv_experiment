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

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.engine import default_argument_parser, default_setup, default_writers, launch
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


logger = logging.getLogger("detectron2")


def get_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = "coco"
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    if len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


def do_test(cfg, model):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name)
        evaluator = get_evaluator(
            cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        )
        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
    if len(results) == 1:
        results = list(results.values())[0]
    return results


def do_train(cfg, model, resume=False):
    model.train()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )
    start_iter = (
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    )
    max_iter = cfg.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = default_writers(cfg.OUTPUT_DIR, max_iter) if comm.is_main_process() else []

    # compared to "train_net.py", we do not support accurate timing and
    # precise BN here, because they are not trivial to implement in a small training loop
    data_loader = build_detection_train_loader(cfg)
    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            storage.iter = iteration

            loss_dict = model(data)
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()

            if (
                cfg.TEST.EVAL_PERIOD > 0
                and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0
                and iteration != max_iter - 1
            ):
                do_test(cfg, model)
                # Compared to "train_net.py", the test results are not dumped to EventStorage
                comm.synchronize()

            if iteration - start_iter > 5 and (
                (iteration + 1) % 20 == 0 or iteration == max_iter - 1
            ):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)

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
#         print(f'video_path = {video_path}, num images = {len(new_dict)}')
        dataset_dict_group += new_dict

    return dataset_dict_group

if __name__ == "__main__":
    wandb.init(project='detectron2', name="plain_train", sync_tensorboard=True)

    d_train = get_dict_for_all_cameras(data_dir='~/data', dataset_group='train')
    d_val = get_dict_for_all_cameras(data_dir='~/data', dataset_group='val')
    print(f'training images = {len(d_train)}, val images = {len(d_val)}')
    TOTAL_NUM_IMAGES = len(d_train)

    classes = ["PERSON", "PIT", "HARDHAT", "SAFETY_VEST"]
    train_val_func = partial(get_dict_for_all_cameras, data_dir='~/data')

    # Registering the Dataset
    for d in ["train", "val"]:
        DatasetCatalog.register("voxel_" + d, lambda d=d: train_val_func(dataset_group=d))
        MetadataCatalog.get("voxel_" + d).set(thing_classes=classes)

    train_metadata = MetadataCatalog.get("voxel_train")
    val_metadata = MetadataCatalog.get("voxel_val")

    epochs = 20

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")) 
    cfg.DATASETS.TRAIN = ("voxel_train",)
    cfg.DATASETS.TEST = ("voxel_val",)
    cfg.DATALOADER.NUM_WORKERS = 4
    # initialize with pre-trained weights
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  
    # cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'model_final.pth' )
    cfg.MODEL.WEIGHTS = "/home/ramin_voxelsafety_com/voxel/experimental/ramin/detectron2/output/model_final.pth"
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

    cfg.SOLVER.STEPS = (int(cfg.SOLVER.MAX_ITER * 0.6),) # reduce LR towards the end

    cfg.TEST.EVAL_PERIOD = batches_per_epoch # do eval once per epoch. 
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 # Set threshold for this model

    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))

    do_train(cfg, model, resume=False)
    do_test(cfg, model)


  