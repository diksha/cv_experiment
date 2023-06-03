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

# constants
API_KEY = "test_5e1c456d97854e39915291f2b809a34a"
DATASET_ID = "ds_c6fb8bhca3rg048yk2t0"

STATE_METADATA = ["open", "closed", "partially_open"]
SPLIT_METADATA = ["test", "val", "train"]
ROOT_DIR = "/data/door_classifier/"


# Create Nucleus client and connect to dataset
client = nucleus.NucleusClient(API_KEY)
dataset = client.get_dataset(DATASET_ID)



# add to loop
for state in STATE_METADATA:
    for split in SPLIT_METADATA:
        dir = os.path.join(ROOT_DIR, split, state)

        print("Directory: ", dir)
        
        filenames = os.listdir(dir)

        for filename in filenames: 
            filepath = os.path.join(dir, filename)
            file_reference_id = os.path.splitext(filename)[0]

            filename_split = file_reference_id.split("_")
            
            padding = filename_split[-1]
            
            if "ch12" in filename_split:
                # Upload image
                dataset_item = nucleus.DatasetItem(
                    image_location=filepath, 
                    reference_id=file_reference_id, 
                    metadata={
                        "state": state, 
                        "split": split,
                        "padding" : padding
                        }
                    )

                response = dataset.append(items=[dataset_item], update=False)
                print("Dataset Response: ", response)

                # Upload label
                category_gt = nucleus.CategoryAnnotation(
                    label=state, 
                    taxonomy_name="door_state", 
                    reference_id=file_reference_id)

                response = dataset.annotate(
                    annotations=[category_gt],
                    update=True,
                    asynchronous=False # async is recommended, but sync jobs are easier to debug
                )
                print("Category Response: ", response)
            

       