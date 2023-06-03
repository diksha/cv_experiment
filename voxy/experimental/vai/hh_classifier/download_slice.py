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
import urllib.request

print(nucleus.__version__)

# constants
API_KEY = "test_5e1c456d97854e39915291f2b809a34a"
DATASET_ID = "ds_c7gxa10da6sg0asdn9bg"

'''
SLICE IDS:
    VALIDATION
        HAT: slc_c7h061gmvh9g05rzk82g
        NO_HAT: slc_c7h05qengdr005r7cqyg
    TRAIN
        HAT: slc_c7h0s8nedg7g0780d6mg
        NO_HAT: slc_c7h0sfxprz9g05rs0pcg
'''
SLICE_ID = "slc_c7h05qengdr005r7cqyg"

STATE_METADATA = ["hat", "no_hat"]
STATE_METADATA = ["hat"]
SITE_METADATA = ["americold", "meijer", "uscold", "formfitness"]
SITE_METADATA = ["uscold",]
ROOT_DIR = "/data/hh/"


# Create Nucleus client and connect to dataset
client = nucleus.NucleusClient(API_KEY)
dataset = client.get_dataset(DATASET_ID)

# Train / Test / Val Split
train_split = 8
test_split = 1
val_split = 1
split_distribution = ['train'] * train_split + ['test'] * test_split + ['val'] * val_split


# slice & directory
data_slice = client.get_slice(SLICE_ID)
split = 'val'
state = 'hat'

dir = os.path.join(ROOT_DIR, split, state)

# Download
results = data_slice.export_raw_items()

for result in results['raw_dataset_items']:
    ref_id = result['ref_id']
    img_url = result['scale_url']
    
    full_path = os.path.join(dir, ref_id)+'.jpg'
    urllib.request.urlretrieve(img_url, full_path)

    print("full path: ", full_path)
       