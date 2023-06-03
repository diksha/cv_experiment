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
import torch
import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
import ffmpeg
import os
import cv2
import numpy as np
from tqdm import tqdm
from core.perception.pose.pose_embedder import KeypointPoseEmbedder
from ray.util.multiprocessing import Pool
import ray
import psutil
from ray.util import ActorPool
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.data.transforms as T

class MultiFrametPredictor:
    def __init__(self, cfg):
        self.cfg = cfg.clone()  
        self.model = build_model(self.cfg)
        self.model.eval()
        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, frames):
        with torch.no_grad():  
            inputs = []
            for original_image in frames:
                height, width = original_image.shape[:2]
                image = self.aug.get_transform(original_image).apply_image(original_image)
                image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

                d = {"image": image, "height": height, "width": width}
                inputs.append(d)

            predictions = self.model(inputs)
            return predictions

@ray.remote(num_gpus=0.1)
class FeatureActor:

    def __init__(self, cuda=True, batch_size=10):
        cfg = get_cfg()
        cfg_101 = "COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml"
        cfg_x101 = "COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml"
        cfg.merge_from_file(model_zoo.get_config_file(cfg_x101))
        #   cfg.merge_from_file(config_path)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model

        cfg.INPUT.MIN_SIZE_TEST = 800
        cfg.INPUT.MAX_SIZE_TEST = 1400

        path_prefix = "/home/ramin_voxelsafety_com/voxel/experimental/ramin/pose_detection/"
        # weight_101 = path_prefix + "data/model_final_997cc7.pkl"
        # weight_x101 = path_prefix + "data/model_final_5ad38f.pkl"

        # cfg.MODEL.WEIGHTS = weight_x101
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml")  
        if cuda == False:
            cfg.MODEL.DEVICE="cpu"

        # self.predictor = DefaultPredictor(cfg)

        # self.predictor = torch.jit.load(path_prefix + "models/model.ts").float().cuda()
        # self.predictor.eval()

        self.predictor = MultiFrametPredictor(cfg)

        self.metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        self.batch_size = batch_size

    def get_features_per_frame(self, frame):

        img_t = torch.as_tensor(frame.astype("float32").transpose(2, 0, 1))
        # img_ = frame.transpose(2, 0, 1)
        # img_ = np.ascontiguousarray(img_)
        # img_t = torch.from_numpy(img_).float().cuda()

        with torch.no_grad():
            pose_output = self.predictor(img_t)
        keypoints_per_frame = pose_output[3].to("cpu").numpy()

        # pose_output = self.predictor(frame)
        # keypoints_per_frame = pose_output["instances"].pred_keypoints.detach().cpu().numpy()

        if keypoints_per_frame.shape[1] != len(self.metadata.keypoint_names):
            print("too few landmarks. skipping this frame", keypoints_per_frame.shape)
            return None
        if keypoints_per_frame.shape[0] > 0:
            keypoints_per_frame = keypoints_per_frame[0, :, :2]
        else:
            print(f"skipped a frame due to no keypoints. shape = {keypoints_per_frame.shape}")
            return None
        feature_creator = KeypointPoseEmbedder.from_landmarks(self.metadata, landmarks=keypoints_per_frame)
        features = feature_creator.create_features()
        return features

    def get_batch_of_frames(self, vcap, rotate_code):
        frames = []
        for k in range(self.batch_size):
            ret, frame = vcap.read()
            if frame is None: continue
            if not ret:
                break
            # check if the frame needs to be rotated
            # if rotate_code is not None:
            #     frame = self.correct_rotation(frame, rotate_code)

            frames.append(frame)
            # also add the horizontal flip for data augmentation
            frames.append(frame[:, ::-1, :])

        return np.array(frames)

    def get_features_per_batch(self, frames):
        pose_outputs = self.predictor(frames)

        feaures_per_batch = []
        person_boxes_per_batch = []
        kp_per_batch = []
        
        for pose_output in pose_outputs:
            keypoints_per_frame = pose_output["instances"].pred_keypoints.detach().cpu().numpy()
            person_boxes = pose_output["instances"].pred_boxes.tensor.detach().cpu().numpy()

            if len(keypoints_per_frame) == 0:
                print('No keypoints in frame. Skipping')
                continue

            feaures_per_frame = None
            for keypoints_per_person in keypoints_per_frame:
                if len(keypoints_per_person) != len(self.metadata.keypoint_names):
                    print('too few landmarks. skipping this frame', keypoints_per_frame.shape)
                    continue

                # remove the confidence
                keypoints_per_person = keypoints_per_person[:, :2]

                feature_creator = KeypointPoseEmbedder(self.metadata, landmarks=keypoints_per_person)
                features_per_person = feature_creator.create_features()
                features_per_person = features_per_person.reshape((1, -1))
                feaures_per_frame = features_per_person if feaures_per_frame is None else np.append(feaures_per_frame, features_per_person, axis=0)
            
            feaures_per_batch.extend(feaures_per_frame)
            person_boxes_per_batch.extend(person_boxes)
            kp_per_batch.extend(keypoints_per_frame)
            
        return feaures_per_batch, person_boxes_per_batch, kp_per_batch  

    def get_features_for_video(self, video_path, visualize=False, max_frames=None):
        print(f'processing {video_path}')
        vcap = cv2.VideoCapture(video_path)
        # check if video requires rotation
        rotate_code = self.check_rotation(video_path)
        print(f'rotate code = {rotate_code}')
        width = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_size = (width,height)
        fps = int(vcap.get(cv2.CAP_PROP_FPS))
        frame_count = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if visualize:
            # fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fourcc = cv2.VideoWriter_fourcc("M","J","P","G")
            fname = video_path.split("/")[-1]
            # fname = pathlib.PurePath(video_path).name
            video_out = cv2.VideoWriter(f"./videos/keypoints_{fname}.avi",
                                        fourcc,
                                        fps, (width, height)
                                    )
        features_all_frames = []
        kp_all_frames = []
        for k in tqdm(range(0, frame_count, self.batch_size)):
            if max_frames is not None and k > max_frames: break
            frames = self.get_batch_of_frames(vcap, rotate_code)
            if len(frames) < self.batch_size: continue

            features_per_batch, _, kp_per_batch = self.get_features_per_batch(frames)
            features_all_frames.extend(features_per_batch)
            kp_all_frames.extend(kp_per_batch)
                        
            if visualize:
                v = Visualizer(frame[:,:,::-1], metadata, scale=1.0)
                v._KEYPOINT_THRESHOLD = 0.02
                out = v.draw_instance_predictions(pose_output["instances"].to("cpu"))
                img_out = out.get_image()
                img_out = np.ascontiguousarray(img_out)
                video_out.write(img_out[:, :, ::-1])
                scores = pose_output["instances"].scores
        vcap.release()
        
        if visualize:
            video_out.release()
        
        return features_all_frames, video_path, kp_all_frames


    def check_rotation(self, path_video_file):
        # this returns meta-data of the video file in form of a dictionary
        meta_dict = ffmpeg.probe(path_video_file)
        # from the dictionary, meta_dict["streams"][0]["tags"]["rotate"] is the key
        # we are looking for
        rotate = meta_dict.get("streams", [dict(tags=dict())])[0].get("tags", dict()).get("rotate", 0)
        rotate = int(rotate)
        rotateCode = None
        if rotate == 90:
            rotateCode = cv2.ROTATE_90_CLOCKWISE
        if rotate == 180:
            rotateCode = cv2.ROTATE_180
        if rotate == 270:
            rotateCode = cv2.ROTATE_90_COUNTERCLOCKWISE
        return rotateCode

    def correct_rotation(self, frame, rotateCode):
        return cv2.rotate(frame, rotateCode)

def get_files(video_path):
    files = [os.path.join(video_path, f) for f in os.listdir(video_path) if os.path.isfile(os.path.join(video_path, f))]
    # files = ['/home/ramin_voxelsafety_com/voxel/experimental/ramin/pose_detection/videos/training/kevin_random.MOV']
    return files

def append_features(list_of_features_per_video_and_fname_and_kp):
    labels = []
    features_all_videos = None
    kp_all_videos = None
    # list_of_features_per_video, file_ = list_of_features_per_video_and_fname
    for features_per_video, fname, kp_per_video in list_of_features_per_video_and_fname_and_kp:
        features_all_videos = features_per_video if features_all_videos is None else np.append(features_all_videos, features_per_video, axis=0)
        kp_all_videos = kp_per_video if kp_all_videos is None else np.append(kp_all_videos, kp_per_video, axis=0)
        gt = None
        if "bad" in fname:
            gt = 0
        if "good" in fname:
            gt = 1
        if "random" in fname:
            gt = 2
        # print(f'features_per_video = {features_per_video}')
        label = [gt]*len(features_per_video)
        labels = np.append(labels, label)
        # labels.extend(label)
        # labels = np.array(labels)

    return features_all_videos, labels, kp_all_videos


if __name__ == "__main__":
    num_cpus = psutil.cpu_count(logical=False)
    ray.init(num_cpus=num_cpus, num_gpus=1)
    
    path_prefix = "/home/ramin_voxelsafety_com/voxel/experimental/ramin/pose_detection/"


    # Create 10 actors
    pool = ActorPool([FeatureActor.remote() for _ in range(1)])

    vid_path_train = path_prefix + "videos/training"
    train_vid_files = get_files(vid_path_train)
    list_of_features_per_video_and_fname = pool.map(lambda actor, file_path: actor.get_features_for_video.remote(file_path), train_vid_files)
    X_train, y_train, keypoints_train = append_features(list_of_features_per_video_and_fname)

    vid_path_test = path_prefix + "videos/testing"
    test_vid_files = get_files(vid_path_test)
    list_of_features_per_video_and_fname = pool.map(lambda actor, file_path: actor.get_features_for_video.remote(file_path), test_vid_files)
    X_test, y_test, keypoints_test = append_features(list_of_features_per_video_and_fname)

    feature_path = path_prefix + "data/features_batched_with_kp.npz"
    np.savez(feature_path, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, keypoints_train=keypoints_train, keypoints_test=keypoints_test)
    print(f"Wrote featutes to {feature_path}")