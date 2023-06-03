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

from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import time
import os


def divide_set(dataset = None):
    num_val= int(len(dataset) * 0.2)
    num_train = len(dataset) - num_val
    train, val = random_split(dataset, [num_train, num_val])
    return train, val

def plot_imgs(imgs, ncol=3, save_dir = None):
    num_row = len(imgs) // ncol # get number of rows depends on the input

    fig, axes = plt.subplots(num_row, ncol, figsize = (ncol * 5, num_row * 5), sharex='all', sharey='all') # create subplot

    for i in range(len(imgs)):
        axes[i // ncol, i % ncol]
        axes[i // ncol, i % ncol].imshow(imgs[i]) # plot each images in the axis
    if save_dir:
        timestr = time.strftime("%Y%m%d-%H%M%S")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir) 
        fig.savefig(os.path.join(save_dir, timestr + '.png'))

def plot_imgs_align(imgs, save_dir = None):
    plot_imgs(np.array(reduce(lambda x,y: x+y, zip(*imgs))), ncol = len(imgs), save_dir=save_dir) # passing the images to be plotted

def input_to_image(input):
    input = input.numpy().transpose((1, 2, 0)) # reshape the input to be like image m by n by 3
    input = np.clip(input, 0, 1) # clip teh values out of [0,1] band
    input = (input * 255).astype(np.uint8) # multiply by 255 (pixels are in range of 0 to 255)
    return input

def masks_to_colorimg(masks, num_class = 2):
    
    colorimg = np.ones((masks.shape[1], masks.shape[2]), dtype=np.float32) * 255 # initialize the output color image
    idx = np.argmax(masks, axis = 0)
    colorimg [idx == 0] = 0
    colorimg [idx == 1] = 1
    colorimg = np.dstack([colorimg]*3)
    return colorimg.astype(np.uint8)
