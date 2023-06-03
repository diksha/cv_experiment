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
import itertools

import numpy as np
import torch
from batchgenerators.augmentations.normalizations import mean_std_normalization

from .utils import convert_to_3d_connected


class Infer:
    def __init__(
        self,
        model,
        crop_shape,
        crop_stride,
        max_hu,
        min_hu,
        mean,
        std,
        device,
        num_classes=3,
        is_hierarchical=False,
    ):
        self.model = model
        self.model.eval()
        self.crop_shape = crop_shape
        self.crop_stride = crop_stride
        self.max_hu = max_hu
        self.min_hu = min_hu
        self.mean = mean
        self.std = std
        self.device = device
        self.num_classes = num_classes
        self.multiple = 8
        self.is_hierarchical = is_hierarchical

    def pad_image(self, img):
        assert isinstance(img, np.ndarray)
        assert len(self.crop_shape) == len(img.shape) - 1
        assert len(self.crop_stride) == len(img.shape) - 1
        # Generate new padded shape using the method (k-1)*stride + crop_shape on each dimension
        # where k is img_shape/stride.
        new_shape = [
            (int(img.shape[d] / self.crop_stride[d]) - 1) * self.crop_stride[d]
            + self.crop_shape[d]
            for d in range(1, len(img.shape))
        ]
        new_shape = [img.shape[0]] + new_shape
        # Determine pad shape for each dimension by subtracting image shape. Pad shape should be before, after.
        # Do floor on before and ceil on after.
        pad_shape = [
            (
                int(np.floor((new_shape[d] - img.shape[d]) / 2)),
                int(np.ceil((new_shape[d] - img.shape[d]) / 2)),
            )
            for d in range(1, len(img.shape))
        ]
        pad_shape = [(0, 0)] + pad_shape
        constant_values = [(self.max_hu, self.max_hu) for d in range(len(img.shape))]
        return (
            np.pad(img, pad_shape, "constant", constant_values=constant_values),
            pad_shape,
        )

    def pad_image_without_crop(self, img):
        # img is KHW.. where we don't want to pad K dimension.
        assert isinstance(img, np.ndarray)

        new_shape = [
            img.shape[d] + (self.multiple - (img.shape[d] % self.multiple))
            for d in range(1, len(img.shape))
        ]
        new_shape = [img.shape[0]] + new_shape
        # Determine pad shape for each dimension by subtracting image shape. Pad shape should be before, after.
        # Do floor on before and ceil on after.
        pad_shape = [
            (
                int(np.floor((new_shape[d] - img.shape[d]) / 2)),
                int(np.ceil((new_shape[d] - img.shape[d]) / 2)),
            )
            for d in range(1, len(img.shape))
        ]
        pad_shape = [(0, 0)] + pad_shape
        constant_values = [(self.max_hu, self.max_hu) for d in range(len(img.shape))]
        return (
            np.pad(img, pad_shape, "constant", constant_values=constant_values),
            pad_shape,
        )

    def unpad_image(self, img, pad_shape):
        index = []
        # Create slice to unpad the array from the pad_shape we used to pad.
        # Use before (0) as it is, for after make in negative to remove that column, for 0 set to None.
        # Because 0:0 is wrong instead it should be 0:None. Use python slice to generate the index for
        # cropping numpy array.
        for d in pad_shape:
            if d[1] == 0:
                index.append(slice(d[0], None))
            else:
                index.append(slice(d[0], -1 * d[1]))
        return img[tuple(index)]

    def unfold_and_fold(self, img):
        # input img is KHW...
        # Create it as cHW image where c is num of classes because model output
        # logits will have this dimension.
        count_image = np.zeros((self.num_classes,) + img.shape[1:])
        sum_image = np.zeros((self.num_classes,) + img.shape[1:])

        # Create all possible combination of [(0, shape), (stride, shape+stride) ... (end-shape, end)] for
        # each dimension and the product them using itertools
        # so we can create all possible overlapping patches in all dimensions.
        for sub_shape in itertools.product(
            *[
                [
                    (i, i + self.crop_shape[d])
                    for i in range(
                        0,
                        img.shape[d] - 1 - self.crop_shape[d] + self.crop_stride[d],
                        self.crop_stride[d],
                    )
                ]
                for d in range(1, len(img.shape))
            ]
        ):

            crop_dimensions = tuple([slice(*i) for i in sub_shape])
            sub_img = img[:, crop_dimensions]
            sub_img_tensor = torch.from_numpy(sub_img)
            # Add the N dimension, batch dimension.
            sub_img_tensor = sub_img_tensor.unsqueeze(dim=0).float()
            batch_logits = self.model.forward(sub_img_tensor.to(self.device))
            if self.is_hierarchical:
                back_fore_img = (
                    torch.nn.Softmax(dim=0)(batch_logits[0][:2, ...])
                    .cpu()
                    .detach()
                    .numpy()
                )
                organ_tumor_pred = (
                    torch.nn.Softmax(dim=0)(batch_logits[0][2:, ...])
                    .cpu()
                    .detach()
                    .numpy()
                )
                pred_sub_img = np.concatenate([back_fore_img, organ_tumor_pred])
            else:
                pred_sub_img = (
                    torch.nn.Softmax(dim=0)(batch_logits[0]).cpu().detach().numpy()
                )

            count_image[(slice(None, None),) + crop_dimensions] += 2.0
            sum_image[(slice(None, None),) + crop_dimensions] += pred_sub_img

        # Get average of logits.
        sum_image = sum_image / count_image
        return sum_image

    def unfold_and_fold_without_crop(self, img):
        with torch.no_grad():
            img_tensor = torch.from_numpy(img).float()
            img_tensor = img_tensor.to(self.device)
            img_tensor.sub_(self.mean).div_(self.std)
            # Make is NKHW.. by adding N as 1.
            img_tensor = img_tensor.unsqueeze(dim=0)
            batch_logits = self.model.forward(img_tensor)
            if self.is_hierarchical:
                back_fore_img = (
                    torch.nn.Softmax(dim=0)(batch_logits[0][:2, ...])
                    .cpu()
                    .detach()
                    .numpy()
                )
                organ_tumor_pred = (
                    torch.nn.Softmax(dim=0)(batch_logits[0][2:, ...])
                    .cpu()
                    .detach()
                    .numpy()
                )
                pred_img = np.concatenate([back_fore_img, organ_tumor_pred])
            else:
                pred_img = (
                    torch.nn.Softmax(dim=0)(batch_logits[0]).cpu().detach().numpy()
                )

            return pred_img

    def process(self, img):
        # img is KHW..
        if self.crop_shape is None:
            img, pad_shape = self.pad_image_without_crop(img)
            img = np.clip(img, self.min_hu, self.max_hu)
            # We returned image added with channels dimension to get raw
            # logits.
            img_with_logits = self.unfold_and_fold_without_crop(img)
            img_with_logits = self.unpad_image(img_with_logits, pad_shape)
            if (
                self.is_hierarchical
            ):  # the image has 4 channels instead of 3, background, foreground, kidney, tumor
                # argmax on back and fore
                back_fore_img = np.argmax(img_with_logits[:2, ...], axis=0)
                # argmax on kidney and tumor
                organ_tumor_pred = np.argmax(img_with_logits[2:, ...], axis=0)
                # update foreground pixels with kidney and tumor
                back_fore_img[back_fore_img > 0] = (
                    organ_tumor_pred[back_fore_img > 0] + 1
                )
                img = back_fore_img
            else:
                img = np.argmax(img_with_logits, axis=0)
            return img, img_with_logits
        else:
            img, pad_shape = self.pad_image(img)
            img = np.clip(img, self.min_hu, self.max_hu)
            img = mean_std_normalization(img, self.mean, self.std)
            # We returned image added with channels dimension to get raw
            # logits.
            img_with_logits = self.unfold_and_fold(img)
            img_with_logits = self.unpad_image(img_with_logits, pad_shape)
            if self.is_hierarchical:
                back_fore_img = np.argmax(img_with_logits[:2, ...], axis=0)
                organ_tumor_pred = np.argmax(img_with_logits[2:, ...], axis=0)
                back_fore_img[back_fore_img > 0] = (
                    organ_tumor_pred[back_fore_img > 0] + 1
                )
                img = back_fore_img
            else:
                # argmax across class dimension which is at axis 0.
                img = np.argmax(sum_image, axis=0)
            return img, img_with_logits

    def params_to_log(self):
        return {
            "crop_shape": self.crop_shape,
            "crop_stride": self.crop_stride,
            "max_hu": self.max_hu,
            "min_hu": self.min_hu,
            "mean": self.mean,
            "std": self.std,
            "num_classes": self.num_classes,
            "model_class": self.model.__class__.__name__,
            "is_hierarchical": self.is_hierarchical,
        }
