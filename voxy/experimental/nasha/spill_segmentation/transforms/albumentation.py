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

import albumentations as A


class AlbumentationsTransforms:
    def __init__(self):
        pass

    def color_transforms(self):
        """Compose the color transforms

        Returns:
            transform(list): A composed transform
        """
        augmented = A.Compose(
            [
                A.OneOf(
                    [
                        A.RandomBrightnessContrast(
                            brightness_limit=[-0.1, 0.1],
                            contrast_limit=[-0.3, 0.3],
                            p=0.7,
                        ),
                        A.ColorJitter(
                            brightness=0.2,
                            contrast=0.2,
                            saturation=0.2,
                            hue=0.2,
                            always_apply=False,
                            p=0.3,
                        ),
                        A.GaussNoise(
                            var_limit=(100, 400.0),
                            mean=70,
                            per_channel=True,
                            always_apply=False,
                            p=0.3,
                        ),
                    ],
                    p=0.5,
                )
            ],
            p=0.5,
        )
        return augmented

    def block_shuffle(self):
        """Composes a transform to shuffle the image along grids

        Returns:
            transform(list): A composed transform of shuffled images
        """
        augmented = A.Compose(
            [
                A.OneOf(
                    [
                        A.RandomGridShuffle(
                            grid=(3, 3), always_apply=False, p=0.8
                        ),
                        A.RandomGridShuffle(
                            grid=(5, 5), always_apply=False, p=0.2
                        ),
                    ],
                    p=0.5,
                )
            ],
            p=0.5,
        )
        return augmented

    def compose(self, transforms_to_compose):
        """Compose transfoms from list

        Args:
            transforms_to_compose (list): A list of composed transforms

        Returns:
            transform(list): Final composed transform
        """
        # combine all augmentations into single pipeline
        result = A.Compose(
            [item for sublist in transforms_to_compose for item in sublist]
        )
        return result
