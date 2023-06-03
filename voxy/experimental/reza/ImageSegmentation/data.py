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

from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
from glob import glob


class FloorDataset(Dataset):
    def __init__(self, img_dir = None, mask_dir = None, num_class = 2, img_width = None, img_height = None):
        """

        """
        self._img_dir = img_dir
        self._mask_dir = mask_dir
        self._names = [os.path.splitext(file)[0] for file in os.listdir(self._img_dir) if not file.startswith('.')]
        self._num_class = num_class
        self._img_width = img_width
        self._img_height = img_height
    
    def _get_image_array(self, pil_img = None):
        """
        """
        pil_img = pil_img.resize((self._img_width, self._img_height), resample = Image.BICUBIC) # resizing the images
        img_array = np.asarray(pil_img) # create numpy array 
        img_array = img_array.transpose((2, 0, 1)) / 255 # reshape numpy array and normalize to be between 0 and 1
        return np.float32(img_array)
    
    def _get_label_array_from_mask(self, pil_img = None):
        """
        """
        pil_img = pil_img.resize((self._img_width, self._img_height))
        img_array = np.array(pil_img)[:,:,0] # create numpy array 
        label_img = np.zeros((img_array.shape[0], img_array.shape[1], self._num_class), dtype=np.uint16)
        label_img[img_array==0,0] = 1
        label_img[img_array==255,1] = 1
        return label_img
    
    def __getitem__(self, index): 
    
        mask_file = glob(self._mask_dir +  self._names[index] + '.*')
        img_file = glob(self._img_dir +  self._names[index] + '.*')
        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {index}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {index}: {img_file}'
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])
        
        assert img.size == mask.size, \
            f'Image and mask {index} should be the same size, but are {img.size} and {mask.size}'

        input_img_data = self._get_image_array(img)
        
        input_label_data = self._get_label_array_from_mask(mask).transpose(2,0,1).astype('float32') 
        
        return [input_img_data, input_label_data] 
    def __len__(self):
        return len(self._names)

class FloorDatasetWL(Dataset):
    def __init__(self, img_dir = None, num_class = 2):
        self._img_dir = img_dir
        self._names = [os.path.splitext(file)[0] for file in os.listdir(self._img_dir) if not file.startswith('.')]
        self._num_class = num_class
        
    def __len__(self):
        return len(self._names)
    
    def _get_image_array(self, pil_img = None):
        pil_img = pil_img.resize((self._img_width, self._img_height), resample = Image.BICUBIC) # resizing the images 
        img_array = np.asarray(pil_img) # create numpy array 
        img_array = img_array.transpose((2, 0, 1)) / 255 # reshape numpy array and normalize to be between 0 and 1
        return img_array
    
    def __getitem__(self, index): 
        img_file = glob(self._img_dir +  self._names[index] + '.*')
        img = Image.open(img_file[0])
        input_img_data = self._get_image_array(img)
        return input_img_data

    

