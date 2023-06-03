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
import nucleus
import random

# constants
API_KEY = "test_5e1c456d97854e39915291f2b809a34a"
DATASET_ID = "ds_c7gxa10da6sg0asdn9bg"

STATE_METADATA = ["hat", "no_hat"]
STATE_METADATA = ["hat"]
SITE_METADATA = ["americold", "meijer", "uscold", "formfitness"]
SITE_METADATA = ["uscold",]
ROOT_DIR = "/data/ppe/"


# Create Nucleus client and connect to dataset
client = nucleus.NucleusClient(API_KEY)
dataset = client.get_dataset(DATASET_ID)

# Train / Test / Val Split
train_split = 8
test_split = 1
val_split = 1
split_distribution = ['train'] * train_split + ['test'] * test_split + ['val'] * val_split

# add to loop
for state in STATE_METADATA:
    for site in SITE_METADATA:
        dir = os.path.join(ROOT_DIR, state, site)

        print("Directory: ", dir)
        
        filenames = os.listdir(dir)

        category_annotations = []
        dataset_items = []
        for filename in filenames: 
            filepath = os.path.join(dir, filename)
            file_reference_id = os.path.splitext(filename)[0]

            # Randomly place in split
            rand_split = random.choice(split_distribution)
                        
            # Upload image
            dataset_item = nucleus.DatasetItem(
                image_location=filepath, 
                reference_id=file_reference_id, 
                metadata={
                    "state": state, 
                    "site": site,
                    "split": rand_split,
                    }
                )
            
            dataset_items.append(dataset_item)


            # Upload label
            category_gt = nucleus.CategoryAnnotation(
                label=state, 
                taxonomy_name="door_state", 
                reference_id=file_reference_id)

            category_annotations.append(category_gt)


        # Upload all at once
        response = dataset.append(items=dataset_items, update=False)
        print("Dataset Response: ", response)
        
        response = dataset.annotate(
            annotations=category_annotations,
            #annotations=[category_gt],
            update=True,
            asynchronous=True # async is recommended, but sync jobs are easier to debug
        )
        print("Category Response: ", response)
            

       