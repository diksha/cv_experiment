import argparse
import os
import random

import cv2
import numpy as np
import wandb
import yaml

from core.infra.cloud.gcs_utils import download_blob, get_files_in_bucket


class DataInspection:
    def __init__(
        self, dataset_log: str, label_gcs_root_dir: str, sample_percentage: int
    ) -> None:
        """_class to demo the labels on top of images for yolo training debugging_
        Args:
            dataset_log (str): _yaml file containing folder paths (video uuids of interest) in gcs_
            label_gcs_root_dir (str): _gcs root dir of the labels_
            sample_percentage (int): _sample percentage to demo_
        """
        with open(dataset_log, encoding="UTF-8") as f:
            self._data = yaml.safe_load(f)
        wandb.init(
            project="dataset_inspection",
            job_type="yolo",
            entity="voxel-wandb",
        )
        self._table_name = "dataset"
        columns = ["Folder", "Frame_name", "Image"]
        self._demo_table = wandb.Table(columns=columns)
        self._extraction_bucket = self._data["extraction_bucket"]
        self._colormap = {0: (255, 0, 0), 1: (0, 0, 255)}
        self._sample_percentage = sample_percentage
        self.label_gcs_root_dir = label_gcs_root_dir
        if not os.path.exists("./temp"):
            os.makedirs("./temp")

    def demo_label(self) -> None:
        """_method for getting files images inside a video uuid of interest and randomly samply from it for demoing_"""
        files_imgs = []
        for train_path in self._data["train"]:
            if "frame_" in train_path:
                files_imgs.append(train_path)
            else:
                files = list(
                    get_files_in_bucket(
                        self._extraction_bucket, prefix=train_path
                    )
                )
                files = [file.name for file in files]
                n_files = len(files)
                n_sample = int((self._sample_percentage / 100) * n_files)
                files = random.sample(files, n_sample)
                idx = train_path.find("images")
                video_uuid = "/".join(
                    train_path[idx + len("images") :].split("/")[1:]
                )
                print(
                    f"processing {len(files)}/{n_files} images for video uuid: {video_uuid}"
                )
                self._process_files_images(files, video_uuid, file_flag=0)
        if len(files_imgs) > 0:
            n_sample = int((self._sample_percentage / 100) * len(files_imgs))
            print(f"processing {n_sample} / {len(files_imgs)} images")
            files_imgs = random.sample(files_imgs, n_sample)
            self._process_files_images(files_imgs, None, file_flag=1)
        wandb.log({self._table_name: self._demo_table})

    def _process_files_images(
        self, images: list, video_uuid: str, file_flag: int
    ) -> None:
        """_method to process each image and demo its label in wandb_
        Args:
            images (list): _list of files to be processed_
            video_uuid (str): _video uuid_
            file_flag (int): _1 if the path is image path, 0 if the path is folder path_
        """
        for file in images:
            status_full_gcs_path = f"gs://{self._extraction_bucket}/{file}"
            idx = file.find("images")
            if file_flag:
                video_uuid = "/".join(
                    file[idx + len("images") :].split("/")[1:-1]
                )
            status_local_img_path = "./temp/temp.jpg"
            download_blob(status_full_gcs_path, status_local_img_path)
            label_path = f"gs://{self.label_gcs_root_dir}/{video_uuid}/{file.split('/')[-1].replace('.jpg', '.txt')}"
            status_local_label_path = "./temp/temp.txt"
            download_blob(label_path, status_local_label_path)
            im = cv2.imread(status_local_img_path)
            if im is None:
                continue
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            with open(status_local_label_path, "r", encoding="UTF-8") as f:
                gt = f.readlines()
            im = self._plot_gt(im, gt)
            row = [video_uuid, file.split("/")[-1], wandb.Image(im)]
            self._demo_table.add_data(*row)
            os.remove(status_local_img_path)
            os.remove(status_local_label_path)

    def _plot_gt(self, im: np.ndarray, gt: list) -> np.ndarray:
        """_method for plotting labels on images_
        Args:
            im (np.ndarray): _image of interest_
            gt (list): _list of all objects (person, PIT)_
        Returns:
            np.ndarray: _imgages with labels overlayed_
        """
        dh, dw, _ = im.shape
        for actor in gt:
            classid, x, y, w, h = map(float, actor.split(" "))
            top_left_x = max(int((x - w / 2) * dw), 0)
            bottom_right_x = min(int((x + w / 2) * dw), dw - 1)
            top_left_y = max(int((y - h / 2) * dh), 0)
            bottom_right_y = min(int((y + h / 2) * dh), dh - 1)
            cv2.rectangle(
                im,
                (top_left_x, top_left_y),
                (bottom_right_x, bottom_right_y),
                self._colormap[classid],
                4,
            )
            im = cv2.putText(
                im,
                f"w={int(w*dw)},h={int(h*dh)}",
                (top_left_x, top_left_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
        return im


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--gcs_label_root", type=str, default=True)
    parser.add_argument("--sample_percentage", type=int, default=50)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data_inspection = DataInspection(
        args.dataset, args.gcs_label_root, args.sample_percentage
    )
    data_inspection.demo_label()
