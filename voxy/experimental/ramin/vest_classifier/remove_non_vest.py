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

import cv2
import os
import numpy as np
import shutil

from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
import time


# In[88]:


vest_path = '/home/ramin_voxelsafety_com/data/vest_patches/no_vest' 
fnames = [os.path.join(vest_path, x) for x in os.listdir(vest_path) if x.endswith('.jpg')]


# In[20]:


def rgb_to_hsv(rgb):
    rgb = np.array(rgb) * 255
    rgb = rgb.astype(np.uint8)
    img = np.ones([1,1,3]).astype(np.uint8)
    img[0, 0] = rgb
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h = hsv[0, 0, 0]
    s = hsv[0, 0, 1]
    return h, s


# In[89]:


def is_vest(fname, rgb_color=[0.84, 0.96, 0.31], thresh=5):
    img = cv2.imread(fname)
#     print(fname, img.shape)
    img = cv2.blur(img, (7, 7))

    h, s = rgb_to_hsv(rgb_color)
    h_min = int(h * 0.6)
    h_max = int(h * 1.4)
    s_min = int(s * 0.6)
    s_max = int(s * 1.4)

    COLOR_MIN = np.array([h_min, s_min, 5],np.uint8)
    COLOR_MAX = np.array([h_max, s_max, 250],np.uint8)

    try:
        hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    except:
        return True

    frame_threshed = cv2.inRange(hsv_img, COLOR_MIN, COLOR_MAX)

    vest_percent = float(cv2.countNonZero(frame_threshed)) / (frame_threshed.shape[0]*frame_threshed.shape[1]) * 100.0 

    print(f'vest percent = {vest_percent}')
    fig, axs = plt.subplots(2)
    axs[0].imshow(frame_threshed)
    axs[1].imshow(img[:, :, ::-1])
    plt.show()
    
    return vest_percent > thresh


# In[ ]:


orange = [0.99, 0.28, 0.28]
orange_1 = [0.9, .4, 0.38]

green = [0.73, 0.96, 0.37]
green_1 = [0.98, 1.0, 0.59]


bad_vest_path = '/home/ramin_voxelsafety_com/data/vest_patches/good_vest'
os.makedirs(bad_vest_path, exist_ok=True)

k = 0
for fname in tqdm(fnames):
    has_orange = is_vest(fname, rgb_color=orange, thresh=5)
    has_green = is_vest(fname, rgb_color=green, thresh=5)
    has_green_1 = is_vest(fname, rgb_color=green_1, thresh=5)
    has_orange_1 = is_vest(fname, rgb_color=orange_1, thresh=5)

    print(k)
    time.sleep(2)
    k += 1

#     if not (has_orange or has_green or has_green_1 or has_orange_1):
#         file_name = fname.split('/')[-1]
#         to_fname = os.path.join(bad_vest_path, file_name)
#         shutil.move(fname, to_fname)
        


# In[84]:



fname = fnames[27]
is_vest(fname, rgb_color=orange, thresh=5)


# In[87]:


frame_threshed

