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

import torch
from data import FloorDataset
from torch.utils.data import DataLoader
from utils import divide_set, input_to_image, plot_imgs_align, masks_to_colorimg
import argparse
import torch.optim as optim
from torch.optim import lr_scheduler
from fastai.vision.learner import create_unet_model
from fastai.vision.models import resnet34
import wandb
import copy
import time
from tqdm import tqdm
from collections import defaultdict
from loss import calc_loss
import numpy as np
from iou import IoU
import os 

IMAGE_WIDTH = 480
IMAGE_HEIGHT = 320



def print_metrics(metrics, epoch_samples, phase):    
    outputs = []
    for k in metrics.keys():
        outputs.append(f"{k}: {metrics[k] / epoch_samples:.4f}")
    phase_metric =  ", ".join(outputs)
    print(f"{phase}: {phase_metric}")    

def train_unet(train_set = None, validation_set = None, batch_size = None, num_epochs = None, device = None):

    config_defaults = {
        'project' : "floor_segmentation", 
        'entity': "voxel-wandb",
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'augment':False,
        'decoder':'resnet34',
        'optimizer': 'adam',
        'learning_rate': 1e-3,
        'scheduler_step': 5,
        'scheduler_gamma': 0.1,
        'dataset':"synthetic"
    }
    
    wandb.init(project =  "floor_segmentation", name = "num_epochs:{}, batch_size:{}, optimizer : {},learning_rate:{}, scheduler_step:{}, 'schedulr_gamma': {}, decoder: {}".format(config_defaults['num_epochs'],
                                                                                                                                                                      config_defaults['batch_size'],
                                                                                                                                                                      config_defaults['optimizer'],
                                                                                                                                                                      config_defaults['learning_rate'],
                                                                                                                                                                      config_defaults['scheduler_step'],
                                                                                                                                                                      config_defaults['scheduler_gamma'],
                                                                                                                                                                      config_defaults['decoder']), entity = "voxel-wandb", config = config_defaults)
    config = wandb.config
    dataloaders = {
        'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8),
        'val': DataLoader(validation_set, batch_size=batch_size, shuffle=False, num_workers=8),
    } 
    model = create_unet_model(resnet34, 2, (IMAGE_HEIGHT, IMAGE_WIDTH), True, n_in=3)
    model = model.to(device) # passing the model to the device (cpu/gpu)
    optimizer = optim.Adam(model.parameters(), lr = config.learning_rate)

    scheduler = lr_scheduler.StepLR(optimizer, step_size = config.scheduler_step,
                    gamma = config.scheduler_gamma)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    model.train() # set the model to train mode
  
    for epoch in range(config.num_epochs):
        
        print(f'Epoch {epoch}/{config.num_epochs - 1}')
        print('-' * 10)
        since = time.time()
        for phase in ['train', 'val']:
            if phase == 'train':
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])  
                model.train()
            else:
                model.eval()
            metrics = defaultdict(float)
            epoch_samples = 0
            with tqdm(dataloaders[phase], unit="batch") as tepoch:
                for inputs, labels in tepoch:
                
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs.type(torch.float32))
                        loss = calc_loss(outputs, labels, metrics)
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    epoch_samples += inputs.size(0)
                    tepoch.set_postfix(loss=loss.item())

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples
            if phase == 'val':
                wandb.log({"epoch":epoch, "val loss" : epoch_loss}, step = epoch)
            if phase == 'train':
                wandb.log({"epoch":epoch, "train loss":epoch_loss}, step = epoch)
        
            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
            else:
                scheduler.step()
        time_elapsed = time.time() - since
        print(f'{time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f"Best val loss: {best_loss:.4f}")
    wandb.run.summary["loss"] = best_loss
    model.load_state_dict(best_model_wts)
    wandb.join()
    return model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    floor_dataset  = FloorDataset(img_dir = args.data_dir + '/images/', mask_dir = args.data_dir + '/annotations/', num_class = 2, img_width = IMAGE_WIDTH, img_height = IMAGE_HEIGHT)
    print(f"Number of images in the dataset {len(floor_dataset)}")
    train, val = divide_set(floor_dataset)
    print(f"Number of images in the train set {len(train)}")
    print(f"Number of images in the val set {len(val)}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = train_unet(train_set = train, validation_set = val, batch_size = args.batch_size, num_epochs = args.epochs, device = device) 
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    save_path = os.path.join(args.model_dir, 'floor_best_model_{}.pth'.format(time.strftime("%Y%m%d-%H%M%S")))
    torch.save(model.state_dict(), save_path) # save model parameters