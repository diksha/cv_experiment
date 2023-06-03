#
# This document may not be reproduced, republished, distributed, transmitted,
# displayed, broadcast or otherwise exploited in any manner without the express
# prior written permission of Voxel Labs, Inc. The receipt or possession of this
# document does not convey any rights to reproduce, disclose, or distribute its
# contents, or to manufacture, use, or sell anything that it may describe, in
# whole or in part.

import torch
import matplotlib.pyplot as plt
from torch.utils.data import WeightedRandomSampler
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
torch.manual_seed(0)
np.random.seed(0)

class MapDataset(Dataset):

    def __init__(self, dataset, map_fn):
        self.dataset = dataset
        self.map = map_fn

    def __getitem__(self, index):
        if self.map:     
            x = self.map(self.dataset[index][0]) 
        else:     
            x = self.dataset[index][0]  # image
        y = self.dataset[index][1]   # label      
        return x, y

    def __len__(self):
        return len(self.dataset)


def get_class_distribution(dataset_obj, idx2class):
    count_dict = {k:0 for k,v in dataset_obj.class_to_idx.items()}
    for element in dataset_obj:
        y_lbl = element[1]
        y_lbl = idx2class[y_lbl]
        count_dict[y_lbl] += 1
            
    return count_dict


def get_class_distribution_loaders(dataloader_obj, dataset_obj, idx2class):
    count_dict = {k:0 for k,v in dataset_obj.class_to_idx.items()}
    
    for _,j in dataloader_obj:
        y_idx = j.item()
        y_lbl = idx2class[y_idx]
        count_dict[str(y_lbl)] += 1
            
    return count_dict

def get_class_target_list_loaders(dataloader_obj, dataset_obj):
    count_dict = {k:0 for k,v in dataset_obj.class_to_idx.items()}
    target = []
    for _,j in dataloader_obj:
        y_idx = j.item()
        target.append(y_idx)
            
    return target


def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def get_data_loader(dataset, percentage, batch_size, weighted_sampling, data_transforms):
    class_names = dataset.classes
    num_train = len(dataset)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(0.2 * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]
    train = torch.utils.data.Subset(dataset, train_idx)
    val = torch.utils.data.Subset(dataset, valid_idx)
    train = MapDataset(train, data_transforms['train'])
    val =  MapDataset(val, data_transforms['val'])
    dataloaders = {             
        'train': DataLoader(train, batch_size=1, shuffle=False, num_workers=8),
        'val': DataLoader(val, batch_size=1, shuffle=False, num_workers=8),
    }
    idx2class = {v: k for k, v in dataset.class_to_idx.items()}
    
    if weighted_sampling:
        target_list_train = torch.tensor(get_class_target_list_loaders(dataloaders['train'], dataset))
        class_count_train = [i for i in get_class_distribution_loaders(dataloaders['train'], dataset, idx2class).values()]
        class_weights_train = 1./torch.tensor(class_count_train, dtype=torch.float) 
        class_weights_train_all = class_weights_train[target_list_train]
        weighted_sampler_train = WeightedRandomSampler(
        weights=class_weights_train_all,
        num_samples=len(class_weights_train_all),
        replacement=True)
        dataloaders = {
        'train': DataLoader(train, batch_size=batch_size, num_workers=8, sampler=weighted_sampler_train),
        'val': DataLoader(val, batch_size=batch_size, num_workers=8),
        }
    else:
        dataloaders = {
        'train': DataLoader(train, batch_size=batch_size, num_workers=8),
        'val': DataLoader(val, batch_size=batch_size, num_workers=8),
        }
    dataset_sizes = {}
    dataset_sizes['train'] = len(train)
    dataset_sizes['val'] = len(val)

    return dataloaders, dataset_sizes, class_names