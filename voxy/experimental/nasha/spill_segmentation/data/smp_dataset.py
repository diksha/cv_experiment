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
from glob import glob

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    def __init__(
        self,
        img_dir=None,
        mask_dir=None,
        names=None,
        transform=None,
        size=None,
    ):
        self._img_dir = img_dir
        self._mask_dir = mask_dir
        self._names = names
        self.transform = transform
        self.size = size

    def _get_image_array(self, pil_img=None):
        """Get numpy image array


        Args:
            pil_img (Image): An image from dataset. Defaults to None.

        Returns:
            np.array: A numpy array of the image
        """
        pil_img = pil_img.resize(self.size, resample=Image.Resampling.BILINEAR)
        img_array = np.asarray(pil_img)
        return img_array

    def _get_mask_array(self, pil_img=None):
        """Get the annotation array

        Shape of the mask should be [batch_size, num_classes, height, width]
        for binary segmentation num_classes = 1

        Args:
            pil_img (Image): Mask Image. Defaults to None.

        Returns:
            np.array: returns numpy array of image masks
        """
        pil_img = pil_img.resize(self.size, resample=Image.Resampling.NEAREST)
        img_array = np.array(pil_img)
        img_array[img_array == 2] = 0
        return img_array

    def __getitem__(self, index):
        """Gets a single

        Args:
            index (int): The index of the image and mask to be fetched

        Returns:
            list: Input image and mask
        """
        mask_file = glob(self._mask_dir + self._names[index] + ".*")
        img_file = glob(self._img_dir + self._names[index] + ".*")
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])

        img_array = self._get_image_array(img)
        mask_array = self._get_mask_array(mask)
        if self.transform is not None:
            augmented = self.transform(image=img_array, mask=mask_array)
            img_array = augmented["image"]
            mask_array = augmented["mask"]

        input_img_data = img_array.transpose((2, 0, 1)).astype("float32")
        input_label_data = np.expand_dims(mask_array, 0).astype("float32")
        return [input_img_data, input_label_data]

    def __len__(self):
        """Get the length of the dataset

        Returns:
            int: length of the dataset
        """
        return len(self._names)
