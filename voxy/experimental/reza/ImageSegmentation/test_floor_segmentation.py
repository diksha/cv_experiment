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

from email.policy import default
import torch
from data import FloorDataset
from torch.utils.data import DataLoader
from utils import input_to_image, plot_imgs_align, masks_to_colorimg
import argparse
from fastai.vision.learner import create_unet_model
from fastai.vision.models import resnet34
import wandb
import numpy as np
from iou import IoU


IMAGE_WIDTH = 480
IMAGE_HEIGHT = 320
segmentation_classes = [
    'non-floor', 'floor']

def labels():
  l = {}
  for i, label in enumerate(segmentation_classes):
    l[i] = label
  return l

def set_model_init(model_dir = None):
    model = create_unet_model(resnet34, 2, (IMAGE_HEIGHT, IMAGE_WIDTH), True, n_in=3)
    model.load_state_dict(torch.load(model_dir))
    return model

def wb_mask(bg_img, pred_mask, true_mask, img_table):
    img_table.add_data(wandb.Image(bg_img, masks={
    "prediction" : {"mask_data" : pred_mask,  "class_labels" : labels()},
    "ground truth" : {"mask_data" : true_mask,  "class_labels" : labels()}}))

    return wandb.Image(bg_img, masks={
    "prediction" : {"mask_data" : pred_mask,  "class_labels" : labels()},
    "ground truth" : {"mask_data" : true_mask,  "class_labels" : labels()}}), img_table


def test_unet(model = None, test_set = None, save_images_dir = True, batch_size = None, device = None):
    
    wandb.init(project =  "floor_segmentation", name = "sample test", entity = "voxel-wandb")

    per_class_mean_accuracy = [] # per class IoU
    mean_accuracy = [] # average IoU
    model.to(device)
    model.eval()  # model is set for the test process
    iteration = 0
    test_dataloaders =  DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8)
    img_table = wandb.Table(columns=['image'])
    ids = 0
    for inputs, labels in test_dataloaders:
        
        inputs = inputs.to(device) # pasing input to the device
        labels = labels.to(device) # passing label to the device
        pred = model(inputs.type(torch.float32)) # passing images to the model and predict

        pred = pred.data.cpu().numpy()  # bring back the prediction from Gpu to cpu 
        if iteration % 25 == 0:

            input_images_rgb = [input_to_image(x) for x in inputs.cpu()] # input rgb images for visualization
            target_masks_rgb = [masks_to_colorimg(x) for x in labels.cpu().numpy()] # mask labels (ground truth) to color image for visualization
            pred_rgb = [masks_to_colorimg(x) for x in pred] # convert the prediction to images for visualization
            plot_imgs_align([input_images_rgb, target_masks_rgb, pred_rgb], save_dir=save_images_dir) # plot images side by side
            #wandb.log({"original image": input_images_rgb[0], "predictions" : pred_rgb[0], "label": target_masks_rgb[0]})
            mask_list = []

            for i in range(len(input_images_rgb)):
                ids = ids + i
                w_b_mask, img_table = wb_mask(input_images_rgb[i], pred_rgb[i][:,:,0], target_masks_rgb[i][:,:,0], img_table)
                mask_list.append(w_b_mask)
            wandb.log({"predictions" : mask_list})
            
        labels = labels.data.cpu().numpy()
        pred_label = [x for x in pred]
        pred_label = np.asarray(pred_label)
        metric = IoU(num_classes = 2)
        metric.add(torch.from_numpy(pred_label), torch.from_numpy(labels)) # calculating IoU based on the library we cited 
        per_class_IoU, mean_IoU  = metric.value() 
        per_class_mean_accuracy.append(per_class_IoU) # append the per class IoU
        mean_accuracy.append(mean_IoU) # append the average Iou
        iteration +=1
    
    table = wandb.Table(columns=['Evaluation', 'Overall', 'non-floor', 'floor'])
    table.add_data('IoU', sum(mean_accuracy)/len(mean_accuracy), np.average(np.array(per_class_mean_accuracy),0)[0], np.average(np.array(per_class_mean_accuracy),0)[1])
    wandb.log({"IoU" : table})
    wandb.log({"Images" : img_table})
    print(f"Test data average IoU: {sum(mean_accuracy)/len(mean_accuracy)}")
    print("-----------------------------")
    print(f"per class (bakcgorund, floor) IoU: {np.average(np.array(per_class_mean_accuracy),0)}")
    wandb.join()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--save_img_dir", type=str, required=False, default = None)
    parser.add_argument("--batch_size", type=int, required=True)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    floor_dataset_test  = FloorDataset(img_dir = args.data_dir + '/images/', mask_dir = args.data_dir + '/annotations/', num_class = 2, img_width = IMAGE_WIDTH, img_height = IMAGE_HEIGHT)
    print(f"Number of images in the dataset {len(floor_dataset_test)}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = set_model_init(model_dir = args.model_dir)
    test_unet(model = model, test_set = floor_dataset_test, save_images_dir = args.save_img_dir, batch_size = args.batch_size, device = device)
