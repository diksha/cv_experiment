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
import json
import os
import pickle
import shutil

import nibabel as nib
import numpy as np
import torch
import torchvision
from skimage import measure
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from .metrics import dice_from_confusion
from .metrics import iou_from_confusion
from .metrics import precision_from_confusion
from .metrics import recall_from_confusion


def convert_to_3d_connected(inp, max_regions=2):
    kt = np.zeros(inp.shape)
    kt[inp == 1] = 1
    kt[inp == 2] = 1

    # Get the all possible connected components
    connected_out = measure.label(kt, background=0)
    # Get all the connected regions
    region = measure.regionprops(connected_out)
    # Find index of max_regions by area in the connected regions list.
    idx = (-np.array([r.area for r in region])).argsort()[:max_regions]

    seg = np.zeros(inp.shape)
    # Convert biggest connected foregrounds to label 1.
    for i in idx:
        seg[connected_out == i + 1] = 1

    # Set tumor to be label 2 in the connected foreground voxels.
    seg[(inp == 2) & (seg == 1)] = 2
    return seg


def load_only_segmentation(cid, data_path="/mnt/disk/kits19_interpolated/data/"):
    case_id = "case_{:05d}".format(cid)
    case_path = os.path.join(data_path, case_id)
    assert os.path.exists(case_path), "Path {} doesn't exists".format(case_path)
    seg = nib.load(os.path.join(case_path, "segmentation.nii.gz"))
    return seg


def get_foreground_indices(cid, dimension_index, data_path):
    seg = load_only_segmentation(cid, data_path)
    foreground_indices = []
    for idx in range(seg.shape[dimension_index]):
        if dimension_index == 0:
            if np.sum(seg[idx, :, :]) != 0:
                foreground_indices.append(idx)
        elif dimension_index == 1:
            if np.sum(seg[:, idx, :]) != 0:
                foreground_indices.append(idx)
        elif dimension_index == 2:
            if np.sum(seg[:, :, idx]) != 0:
                foreground_indices.append(idx)
    return foreground_indices


def create_metdata_json(data_path, filepath="metadata.json", case_ids=range(0, 210)):
    """
    create_metdata_json("/home/aladdha/medical_imaging/kits19_interpolated_data", "/home/aladdha/medical_imaging/challenges/kits19/metadata.json")

    """
    data = {
        case_id: {
            "shape": load_only_segmentation(case_id, data_path).shape,
            "foreground_indices": {
                "axial": get_foreground_indices(case_id, 0, data_path),
                "coronal": get_foreground_indices(case_id, 1, data_path),
                "sagittal": get_foreground_indices(case_id, 2, data_path),
            },
        }
        for case_id in case_ids
    }

    with open(filepath, "w") as outfile:
        json.dump(data, outfile)


def visualize_numpy_features_as_tensor_image(X):
    to_img = torchvision.transforms.transforms.ToPILImage()
    a = torch.DoubleTensor(X)
    a = a.type(torch.FloatTensor)
    return to_img(a)


def visualize_tensor_as_image(X):
    X = np.squeeze(X)
    return visualize_numpy_features_as_tensor_image(X)


def load_data_indices(filepath="data_split.json"):
    with open(filepath) as json_file:
        data = json.load(json_file)
    return data


def get_cuddn_version():
    return torch.backends.cudnn.version()


def is_cuda_installed():
    return torch.cuda.is_available()


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def write_test_and_cv_splits(
    n_samples=210, test_size=0.1, n_splits=4, filepath="data_split.json"
):
    data = np.random.randn(n_samples, 10)
    labels = np.random.randint(2, size=n_samples)
    indices = np.arange(n_samples)
    x1, x2, y1, y2, idx1, idx2 = train_test_split(
        data, labels, indices, test_size=test_size
    )

    train = idx1
    test = idx2

    data = {"all": {"train": list(map(int, idx1)), "test": list(map(int, idx2))}}

    X = np.array(train)
    y = np.array(range(len(train)))
    rs = KFold(n_splits=n_splits, shuffle=True)

    for n, (train_index, test_index) in enumerate(rs.split(X)):
        data["split{}".format(n)] = {
            "train": list(map(int, X[train_index])),
            "val": list(map(int, X[test_index])),
        }

    with open(filepath, "w") as outfile:
        json.dump(data, outfile)


def generate_affine_header_pickle(data_path, cid, output_path):
    """
    for cid in range(0, 210):
        print(cid)
        generate_affine_header_pickle("/mnt/disk/kits19_interpolated/data", cid, "/mnt/disk2/interpolated_data")
    """
    case_path = os.path.join(data_path, "case_{:05d}".format(cid))
    assert os.path.exists(case_path), "Path {} doesn't exists".format(case_path)
    vol = nib.load(os.path.join(case_path, "imaging.nii.gz"))
    with open(
        os.path.join(output_path, "case_{:05d}".format(cid), "header_affine.pkl"), "wb"
    ) as fp:
        pickle.dump([vol.header, vol.affine], fp)


def convert_data(data_path, cid, output_path):
    """
    for cid in range(0, 210):
        print(cid)
        convert_data("~/medical_imaging/kits19/data", cid, "~/medical_imaging/kits19/kits19_interpolated_data")
    """
    case_path = os.path.join(data_path, "case_{:05d}".format(cid))
    output_path = os.path.join(output_path, "case_{:05d}".format(cid))
    os.makedirs(output_path, exist_ok=True)
    assert os.path.exists(case_path), "Path {} doesn't exists".format(case_path)
    vol = nib.load(os.path.join(case_path, "imaging.nii.gz"))
    np.save(os.path.join(output_path, "vol.npy"), vol.get_data())
    np.save(os.path.join(output_path, "seg.npy"), seg.get_data().astype(np.uint8))
    with open(os.path.join(output_path, "header_affine.pkl"), "wb") as fp:
        pickle.dump([vol.header, vol.affine], fp)
    shutil.copyfile(
        os.path.join(case_path, "segmentation.nii.gz"),
        os.path.join(output_path, "segmentation.nii.gz"),
    )


def infer_fn(device, infer_iter, start_train_iter, max_train_iter):
    """
    from joblib import Parallel, delayed
    jobs = (
            delayed(infer_fn)('cuda:0', 10000, 109998, 110002),
            delayed(infer_fn)('cuda:1', 10000, 129998, 130002),
            delayed(infer_fn)('cuda:3', 10000, 149998, 150002),
            delayed(infer_fn)('cuda:4', 10000, 169998, 170002),
            delayed(infer_fn)('cuda:5', 10000, 189998, 190002)
    )

    Parallel(n_jobs=5, verbose=100)(jobs)
    """
    import json
    import sys

    sys.path.append("/home/aladdha/medical_imaging/challenges/")
    from kits19.train import Train

    x = json.load(
        open(
            "/home/aladdha/medical_imaging/experiments/axial_focal_loss_2d_res_unet_v2/config.json"
        )
    )

    x["device"] = device
    x["train"]["infer_iter"] = infer_iter
    x["train"]["start_train_iter"] = start_train_iter
    x["train"]["max_train_iter"] = max_train_iter
    x["infer"]["train_dataset"]["case_ids"]["indices"]["step"] = 5
    x["infer"]["validation_dataset"]["case_ids"]["indices"]["step"] = None
    x["infer"]["params"]["crop_shape"] = None
    x["infer"]["params"]["crop_stride"] = None

    t = Train(
        config_path="/home/aladdha/medical_imaging/experiments/axial_focal_loss_2d_res_unet_v2/config.json",
        config_overrides=x,
    )
    t.infer()
    print("Complete")


def get_result(data):
    averaged_scores = {}
    for class_id in [0, 1, 2]:
        dice_scores = []
        precisions = []
        recalls = []
        ious = []
        keysaa = []
        for key, val in data.items():
            if key == "params":
                continue
            keysaa.append(key)
            dice_scores.append(val[class_id]["dice"])
            precisions.append(val[class_id]["precision"])
            recalls.append(val[class_id]["recall"])
            ious.append(val[class_id]["iou"])

        bottom_k = np.argsort(np.array(dice_scores))[:40]
        top_k = np.argsort(-np.array(dice_scores))[:40]
        cidsaa = np.array(keysaa)

        averaged_scores[class_id] = {
            "dice": sum(dice_scores) / len(dice_scores),
            "precision": sum(precisions) / len(precisions),
            "recall": sum(recalls) / len(recalls),
            "iou": sum(ious) / len(ious),
            "top_k": " ".join(map(str, cidsaa[top_k])),
            "top_k_scores": ", ".join(
                map(str, np.round(np.array(dice_scores)[top_k], 2))
            ),
            "bottom_k": " ".join(map(str, cidsaa[bottom_k])),
            "bottom_k_scores": ", ".join(
                map(str, np.round((np.array(dice_scores)[bottom_k]), 2))
            ),
        }

    dice_scores = []
    precisions = []
    recalls = []
    ious = []
    for key, val in data.items():
        if key == "params":
            continue
        TP = (
            val["confusion_multiclass"][1][1]
            + val["confusion_multiclass"][2][2]
            + val["confusion_multiclass"][1][2]
            + val["confusion_multiclass"][2][1]
        )
        FP = val["confusion_multiclass"][0][1] + val["confusion_multiclass"][0][2]
        FN = val["confusion_multiclass"][1][0] + val["confusion_multiclass"][1][2]
        TN = val["confusion_multiclass"][0][0]
        dice_scores.append(dice_from_confusion(TP, TN, FP, FN))
        precisions.append(precision_from_confusion(TP, TN, FP, FN))
        recalls.append(recall_from_confusion(TP, TN, FP, FN))
        ious.append(iou_from_confusion(TP, TN, FP, FN))

    averaged_scores[4] = {
        "dice": sum(dice_scores) / len(dice_scores),
        "precision": sum(precisions) / len(precisions),
        "recall": sum(recalls) / len(recalls),
        "iou": sum(ious) / len(ious),
    }

    conf = np.zeros((3, 3))
    for key, val in data.items():
        if key == "params":
            continue
        conf += np.array(val["confusion_multiclass"])

    averaged_scores["conf"] = list(map(list, conf))
    return averaged_scores


def get_result_from_path(path):
    data = json.load(open(path))
    return get_result(data)
