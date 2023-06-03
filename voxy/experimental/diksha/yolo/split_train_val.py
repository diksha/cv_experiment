import tempfile

import pandas as pd
from sklearn.model_selection import train_test_split

from core.utils.aws_utils import download_to_file


def split_train_val():
    """Split training and validation set for lightly downsampled images"""
    train_uuids = []
    test_uuids = []
    with tempfile.NamedTemporaryFile() as temp_file:
        download_to_file(
            "voxel-users", "diksha/training/yolo_sampeled_set", temp_file.name
        )
        dataframe = pd.read_csv(temp_file.name)
    dataframe["location"] = (
        dataframe.iloc[:, 0].str.split("/").str[:7].str.join("/")
    )
    for _, df_group in dataframe.groupby("location"):
        try:
            train, test = train_test_split(df_group, test_size=0.2)
            train_uuids.extend(train[train.columns[0]].values.tolist())
            test_uuids.extend(test[test.columns[0]].values.tolist())
        except Exception:  # trunk-ignore(pylint/W0703)
            train_uuids.extend(df_group[df_group.columns[0]].values.tolist())

    with open(
        "/home/diksha/yolo_sampling/training.txt", "w", encoding="utf-8"
    ) as outfile:
        for line in train_uuids:
            outfile.write(f"{line}\n")

    with open(
        "/home/diksha/yolo_sampling/validation.txt", "w", encoding="utf-8"
    ) as outfile:
        for line in test_uuids:
            outfile.write(f"{line}\n")


split_train_val()
