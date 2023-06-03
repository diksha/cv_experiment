import os
import shutil

from tqdm import tqdm


def get_lightly_filtered_imgs(
    base_path="/home/vivek/lightly_data/",
    dst_path="/home/vivek/voxel/experimental/vivek/lightly_downsampled_30k_hh_v11/",
):
    """Copy images mentioned by the lightly filtered list
    and move to the dst_path folder

    Args:
        base_path (str, optional): prefix for lightly images path.
        Defaults to "/home/vivek/lightly_data/".
        dst_path (str, optional): new prefix since shutil only copys the file to the final dir.
        Defaults to "/home/vivek/voxel/experimental/vivek/lightly_downsampled_30k_hh_v11/".
    """

    with open(
        "/home/vivek/voxel/lightly_10_epochs_r50_30k_quakertownv11/"
        + "2022-11-05/04:27:13/filenames/sampled_filenames_excluding_datapool.txt",
        "r",
        encoding="utf-8",
    ) as image_folder:
        image_folder = list(image_folder)
        for file in tqdm(image_folder):

            file = file.replace("\n", "")
            rev_file = file[::-1]
            idx = rev_file.find("/")
            dir_path = file[: len(file) - idx]

            # print(base_path + file)
            # print(dst_path + dir_path)

            os.makedirs(os.path.dirname(dst_path + dir_path), exist_ok=True)
            shutil.copy(base_path + file, dst_path + dir_path)

    shutil.rmtree(base_path)


get_lightly_filtered_imgs()
