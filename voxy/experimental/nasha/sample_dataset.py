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
import random
import pandas as pd

ROOT_DIR = '/home/nasha_voxelsafety_com/voxel/experimental/nasha/data/hat_classifier/'

for klass in os.listdir(ROOT_DIR):
    sourcepath = os.path.join(ROOT_DIR,klass)
    
    for sourcename in os.listdir(sourcepath):
        imagenames = os.listdir(os.path.join(sourcepath,sourcename))
        
        if sourcename == 'uscold' and klass == 'hat':
            #select only 10% of the data
            im_train = random.sample(imagenames,2500)

            
        elif sourcename == 'original' or sourcename == 'meijer':
            #select all the data
            im_train = imagenames
        else:
            continue

        #write this to csv with the class names as
        df = pd.DataFrame(data= im_train,columns=['image'])
        
        df['image'] =  os.path.join(sourcepath,sourcename) + "/" + df['image'].astype(str)
        df = df.assign(label=klass)

        df.to_csv(path_or_buf=os.path.join(ROOT_DIR,'train.csv'), mode = 'a',index=False,header=False)