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
from typing import Dict, List, Tuple
import torch
from torch import Tensor, nn

import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader, detection_utils
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, print_csv_format
from detectron2.export import (
    Caffe2Tracer,
    TracingAdapter,
    add_export_config,
    dump_torchscript_IR,
    scripting_with_instances,
)
from detectron2.modeling import GeneralizedRCNN, RetinaNet, build_model
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.projects.point_rend import add_pointrend_config
from detectron2.structures import Boxes
from detectron2.utils.env import TORCH_VERSION
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger



def export_tracing(torch_model, inputs):
    assert TORCH_VERSION >= (1, 8)
    image = inputs[0]["image"]
    inputs = [{"image": image}]  # remove other unused keys

    if isinstance(torch_model, GeneralizedRCNN):

        def inference(model, inputs):
            # use do_postprocess=False so it returns ROI mask
            inst = model.inference(inputs, do_postprocess=False)[0]
            return [{"instances": inst}]

    else:
        inference = None  # assume that we just call the model directly

    traceable_model = TracingAdapter(torch_model, inputs, inference)

    ts_model = torch.jit.trace(traceable_model, (image,))
    with PathManager.open(os.path.join("../models", "model.ts"), "wb") as f:
        torch.jit.save(ts_model, f)
    dump_torchscript_IR(ts_model, "../models")
    print("Inputs schema: " + str(traceable_model.inputs_schema))
    print("Outputs schema: " + str(traceable_model.outputs_schema))

    if not isinstance(torch_model, (GeneralizedRCNN, RetinaNet)):
        return None

    def eval_wrapper(inputs):
        """
        The exported model does not contain the final resize step, which is typically
        unused in deployment but needed for evaluation. We add it manually here.
        """
        input = inputs[0]
        instances = traceable_model.outputs_schema(ts_model(input["image"]))[0]["instances"]
        postprocessed = detector_postprocess(instances, input["height"], input["width"])
        return [{"instances": postprocessed}]

    return eval_wrapper




import cv2
img = cv2.imread('./data/two_person_test.png')
import numpy as np

original_image = cv2.resize(img, (1024, 608))

height, width = original_image.shape[:2]
image = torch.as_tensor(original_image.astype("float32").transpose(2, 0, 1))
sample_inputs = [{"image": image, "height": height, "width": width}]



import detectron2 
from detectron2.config import get_cfg
from detectron2 import model_zoo


def init_pose_cfg(cuda=True):
  cfg = get_cfg()
  cfg_101 = 'COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml'
  cfg_x101 = 'COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml'
  cfg.merge_from_file(model_zoo.get_config_file(cfg_101))
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  
   
  cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml")  

  if cuda == False:
      cfg.MODEL.DEVICE='cpu'

  return cfg


# # Save model

from torch import Tensor, nn
from detectron2.export import dump_torchscript_IR, scripting_with_instances
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
import os

os.makedirs('../models', exist_ok=True)
cfg_l = init_pose_cfg()
my_model = build_model(cfg_l)  
# DetectionCheckpointer(my_model).load('/home/ramin_voxelsafety_com/voxel/experimental/ramin/pose_detection/data/model_final_5ad38f.pkl')
DetectionCheckpointer(my_model).load('/home/ramin_voxelsafety_com/voxel/experimental/ramin/pose_detection/data/model_final_997cc7.pkl')

my_model.eval()
exported_model = export_tracing(my_model, sample_inputs)


# # Load model

import torch
import torchvision

script_model = torch.jit.load("../models/model.ts").float().cuda()

import numpy as np
from IPython.display import display, Image

fname = './data/two_person_test.png'
img = cv2.imread(fname)
display(Image(filename=fname))

img = cv2.resize(img, (1024, 608))
img_ = img.transpose(2, 0, 1)
img_ = np.ascontiguousarray(img_)
img_t = torch.from_numpy(img_).float().cuda()


print(img_t.shape)
with torch.no_grad():
    out = script_model(img_t)


# # Visualize


from detectron2.utils.visualizer import Visualizer
from matplotlib import pyplot as plt

v = Visualizer(img[:, :, ::-1],
                # metadata=val_metadata, 
                scale=2.0, 
)
out_v = v.overlay_instances(keypoints=out[3].to("cpu"))
# out_v = v.draw_and_connect_keypoints(out[3].to("cpu"))
img_out = out_v.get_image()
img_out = np.ascontiguousarray(img_out)

plt.imshow(img_out)
plt.show()






