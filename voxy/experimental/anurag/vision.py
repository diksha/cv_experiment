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
import csv
import os
import random

train = [
    "e_dock_north_ch22_20201104000304_20201104040543",
    "e_dock_north_ch4_20201026015837_20201026060339",
    "e_dock_north_ch22_20201104000155_20201104040203",
    "e_dock_north_ch4_20201105000345_20201105040238",
]
val = [
    "e_dock_north_ch12_20201102001829_20201102041946",
    "e_dock_north_ch22_20201105185845_20201105200042",
]


results = []

for vid in train:
    files = [
        f
        for f in os.listdir(
            os.path.join("/home/anurag_voxelsafety_com/data", vid, "labels")
        )
        if f.endswith(".txt")
    ]
    for fname in files:
        data = open(
            os.path.join("/home/anurag_voxelsafety_com/data", vid, "labels", fname)
        ).readlines()
        for item in data:
            item = item.replace("\n", "").split(" ")
            kls, xc, yc, w, h = (
                item[0],
                float(item[1]),
                float(item[2]),
                float(item[3]),
                float(item[4]),
            )
            x1 = xc - w / 2.0
            y1 = yc - h / 2.0
            x2 = x1 + w
            y2 = y1 + h
            if y2 > 1:
                y2 = 1.0
            if x2 > 1:
                x2 = 1.0
            if x1 < 0:
                x1 = 0.0
            if y1 < 0:
                y1 = 0.0
            results.append(
                [
                    "TRAIN",
                    f"gs://voxel-users/anurag/yolo/{vid}/images/{fname.replace('.txt', '.jpg')}",
                    "PIT",
                    x1,
                    y1,
                    x2,
                    y1,
                    x2,
                    y2,
                    x1,
                    y2,
                ]
            )

for vid in val:
    files = [
        f
        for f in os.listdir(
            os.path.join("/home/anurag_voxelsafety_com/data", vid, "labels")
        )
        if f.endswith(".txt")
    ]
    for fname in files:
        data = open(
            os.path.join("/home/anurag_voxelsafety_com/data", vid, "labels", fname)
        ).readlines()
        name = random.choice(["TEST", "VALIDATE"])
        for item in data:
            item = item.replace("\n", "").split(" ")
            kls, xc, yc, w, h = (
                item[0],
                float(item[1]),
                float(item[2]),
                float(item[3]),
                float(item[4]),
            )
            x1 = xc - w / 2.0
            y1 = yc - h / 2.0
            x2 = x1 + w
            y2 = y1 + h
            if y2 > 1:
                y2 = 1.0
            if x2 > 1:
                x2 = 1.0
            if x1 < 0:
                x1 = 0.0
            if y1 < 0:
                y1 = 0.0
            results.append(
                [
                    name,
                    f"gs://voxel-users/anurag/yolo/{vid}/images/{fname.replace('.txt', '.jpg')}",
                    "PIT",
                    x1,
                    y1,
                    x2,
                    y1,
                    x2,
                    y2,
                    x1,
                    y2,
                ]
            )

with open("/home/anurag_voxelsafety_com/data/vision.csv", "w") as f:
    write = csv.writer(f)
    write.writerows(results)
