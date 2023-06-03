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
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import WeightedRandomSampler
import numpy as np
import torchvision
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import time
import os
import copy
from autoaugment.autoaugment import ImageNetPolicy
import wandb
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score

def build_dataset(batch_size=4,AUTOAUGMENT = True, data_dir =""):
    
    if AUTOAUGMENT:

        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop((224,224),scale=(0.7,1.0)),
                transforms.RandomHorizontalFlip(),
                ImageNetPolicy(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
                

            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
        }
    else:
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop((224,224),scale=(0.7,1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
    
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                    for x in ['train', 'val']}

    weighted_samplers = {}
    for x in ['train', 'val']:
        target = np.array(image_datasets[x])[:,1]

        class_sample_count = np.array(
            [len(np.where(target == t)[0]) for t in np.unique(target)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in target])

        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()
        weighted_sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        weighted_samplers[x] = weighted_sampler

    
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                    num_workers=1, sampler=weighted_samplers[x])
                for x in ['train', 'val']}
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    return dataloaders, dataset_sizes

def train_init(freeze_layers=True,n_classes=3):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_conv = torchvision.models.resnet50(pretrained=True)

    for param in model_conv.parameters():
        param.requires_grad = not(freeze_layers)

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, n_classes)

    model_conv = model_conv.to(device)

    criterion = nn.CrossEntropyLoss()

    return model_conv, criterion, device

def calculate_metrics(pred, target, threshold=0.5):

    pred = np.array(pred > threshold, dtype=float)

    pred_flat = np.argmax(pred, 1)

    return {'weighted/precision': precision_score(y_true=target, y_pred=pred_flat, average='weighted'),

            'weighted/recall': recall_score(y_true=target, y_pred=pred_flat, average='weighted'),

            'weighted/f1': f1_score(y_true=target, y_pred=pred_flat, average='weighted'),

            }

def train_model(root_dir):
    
    config_defaults = {
        'project' : "door_resnet",
        'num_epochs': 25,
        'batch_size': 128,
        'learning_rate': 1e-3,
        'step':7,
        'augment':True,
        'freeze_layers':True
    }
    
    wandb.login(key='5a3ade2f4b7473a2e9c8a463138793a6e0c7e548')
    wandb.init(config=config_defaults)
    config = wandb.config
   
    model,criterion, device = train_init(config.freeze_layers)
    
    model_path = os.path.join(root_dir,"models")
    # Observe that only parameters of final layer are being optimized as
    # opposed to before.
    optimizer = optim.SGD(model.fc.parameters(), lr=config.learning_rate, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    scheduler = lr_scheduler.StepLR(optimizer, step_size=config.step, gamma=0.1)

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    
    dataloaders, dataset_sizes = build_dataset(batch_size=config.batch_size,AUTOAUGMENT=config.augment,data_dir=root_dir)
    for epoch in range(config.num_epochs):
        print('Epoch {}/{}'.format(epoch, config.num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            model_result = []
            targets = []
            
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    #_, preds = torch.max(outputs, 1)
                    
                    loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Metrics
                running_loss += loss.item() * inputs.size(0)

                model_result.extend(outputs.cpu().detach().numpy())
                targets.extend(labels.cpu().numpy())

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]


            print("Model Result: ", np.array(model_result).shape, " Targets: ", np.array(targets).shape)
            
            results = calculate_metrics(np.array(model_result),np.array(targets))
            if phase == 'val':
                wandb.log({"val loss":epoch_loss})
                wandb.log({"val_f1":results['weighted/f1']})
            if phase == 'train':
                wandb.log({"train loss":epoch_loss})
            
            print('{} Loss: {:.4f} Precision: {:.4f} Recall: {:.4f} F1: {:.4f}'.format(
                phase, epoch_loss, results['weighted/precision'],results['weighted/recall'], results['weighted/f1']))

            # deep copy the model
            if phase == 'val': 
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    
    # load best model weights
    model.load_state_dict(best_model_wts)

    #save this model
    example_input = torch.randn(1, 3, 224, 224, requires_grad=False).to(device)
    traced_model = torch.jit.trace(model, example_input)
    today_date = datetime.today().strftime('%Y-%m-%d')
    traced_model.save(root_dir + f'/door_classifier_resnet50_{today_date}.pth')

    # TODO: (vai) use wandb to store the best model
    wandb.join()
    return model

if __name__ == "__main__":
    root_dir = "/data/"
    root_dir = "/data/door_classifier_americold"

    train_model(root_dir)
