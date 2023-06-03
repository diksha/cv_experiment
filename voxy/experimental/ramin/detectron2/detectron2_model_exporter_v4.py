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


# In[2]:


model_name = '9-10-2021/model_final'


# In[3]:


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
    with PathManager.open(os.path.join("../output", f"{model_name}.ts"), "wb") as f:
        torch.jit.save(ts_model, f)
    dump_torchscript_IR(ts_model, "../output")
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


# In[4]:


import cv2
# from google.colab.patches import cv2_imshow
img = cv2.imread('test_img.png')
import numpy as np

original_image = cv2.resize(img, (1024, 608))
# cv2_imshow(img)

height, width = original_image.shape[:2]
image = torch.as_tensor(original_image.astype("float32").transpose(2, 0, 1))
sample_inputs = [{"image": image, "height": height, "width": width}]


# In[5]:


import detectron2 
from detectron2.config import get_cfg
from detectron2 import model_zoo

model_path = f'../output/{model_name}.pth'

classes = ["PERSON", "PIT", "HARDHAT", "SAFETY_VEST"]
cfg_l = get_cfg()
cfg_l.merge_from_file(model_zoo.get_config_file('COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml'))
cfg_l.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 # Set threshold for this model
cfg_l.MODEL.WEIGHTS = model_name # Set path model .pth
cfg_l.MODEL.ROI_HEADS.NUM_CLASSES = len(classes)


# In[6]:


cfg_l.MODEL.ANCHOR_GENERATOR.SIZES = [[16, 32], [32, 64], [64, 128], [128, 256], [256, 512]]
cfg_l.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]


# In[7]:


from torch import Tensor, nn
from detectron2.export import dump_torchscript_IR, scripting_with_instances
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model

my_model = build_model(cfg_l)  # returns a torch.nn.Module
DetectionCheckpointer(my_model).load(model_path)

my_model.eval()

exported_model = export_tracing(my_model, sample_inputs)


# In[10]:


import torch
import torchvision

script_model = torch.jit.load(f"../output/{model_name}.ts").float().cuda()


# In[11]:


input_image = sample_inputs[0]['image']
print(input_image.shape)
with torch.no_grad():
    out = script_model(input_image)


# In[12]:


with torch.no_grad():
    out_model = my_model(sample_inputs)
    out_model[0]['instances']


# In[13]:


# out


# In[14]:


from detectron2.utils.visualizer import Visualizer
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure


v = Visualizer(original_image[:, :, ::-1],
                # metadata=val_metadata, 
                scale=2.0, 
)

out_v = v.draw_instance_predictions(out_model[0]['instances'].to("cpu"))
img_out = out_v.get_image()
img_out = np.ascontiguousarray(img_out)

# cv2.imwrite('result.png', img_out[:, :, ::-1])
figure(figsize=(25, 10), dpi=80)

plt.imshow(img_out)
plt.show()


# In[ ]:




