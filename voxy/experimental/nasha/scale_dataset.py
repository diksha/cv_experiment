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
from nucleus import NucleusClient, CategoryAnnotation, CategoryPrediction, BoxAnnotation
from nucleus import DatasetItem
import torch
import torchvision
import cv2
import numpy as np
import pickle
import sklearn
import os


class ScaleDataset():

    def __init__(self):
        self.client = NucleusClient('test_5e1c456d97854e39915291f2b809a34a')
        self.dataset_id = None

    def create_dataset(self, dataset_name : str ="Vest Classification Dataset") -> None:
        client = self.client
        self.dataset = client.create_dataset(dataset_name)
        self.dataset_id = self.dataset.id

    def get_dataset(self, dataset_id : str ="ds_c7f290cs7p6008sbn1t0") -> None:
        client = self.client 
        self.dataset = client.get_dataset(self.dataset_id) if self.dataset_id else client.get_dataset(dataset_id)
        if self.dataset_id is None:
            self.dataset_id = self.dataset.id
    def upload_to_dataset(self, image_name, source, ref_id, pose_embedding= None):
        
        metadata = {'source':source}
        if pose_embedding is not None:
            pose_embedding = str(pose_embedding)
            metadata['pose_embedding'] = pose_embedding
        item = DatasetItem(image_location=f"{image_name}", reference_id=ref_id, metadata=metadata)
        print("Item:",item)

        dataset = self.dataset
        
        # after creating or retrieving a Dataset
        job = dataset.append(
            items=[item],
            update=True, 
            asynchronous=False 
        )
    
    def create_taxonomy(self, labels = ["good_lift", "bad_lift","random"], tax_name=None):
        
        dataset = self.dataset
        dataset.add_taxonomy(
        taxonomy_name=tax_name,
        taxonomy_type="category", 
        labels=labels
    )

    def add_gt_category(self,image_name, label, tax_name):

        category_gt = CategoryAnnotation(
        label=label, 
        taxonomy_name=tax_name, 
        reference_id=os.path.basename(image_name), 
        metadata={}
        )

        dataset = self.dataset
        
        dataset.annotate(
        annotations=[category_gt],
        update=True,
        asynchronous=False 
        )

    def add_gt_detection(self,image_name, label, rect_ppe, data_source):

        box_gt = BoxAnnotation(
        label=label,
        x=rect_ppe.top_left_vertice.x,
        y=rect_ppe.top_left_vertice.y,
        width=rect_ppe.w,
        height=rect_ppe.h,
        reference_id=os.path.basename(image_name),
        annotation_id=f"{label}_{image_name}",
        metadata={"source": data_source}
        )

        dataset = self.dataset
        
        job = dataset.annotate(
        annotations=[box_gt],
        update=True,
        asynchronous=False 
        )

        print("Job Done:",job)

    def add_model(self, model_name ="Vest Classifier Incidents", model_ref = "vest-classifier-incidents", metadata= {}):
        #use metadata to version models for now
        client = self.client
        self.model = client.add_model(
        name=model_name,
        reference_id=model_ref,
        metadata=metadata
        )


    def get_model(self, model_id = 'prj_c6jgdv7bwck00skqn31g'):
        client = self.client
        self.model = client.get_model(model_id=model_id)

    def create_slice(self,ref_ids = ["interesting_item_1", "interesting_item_2"], slice_name = "interesting"):
        slice = self.dataset.create_slice(name=slice_name, reference_ids=ref_ids)
        return slice

    def get_slice_urls(self, slice):
        return slice.export_raw_items()

    def run_image_classifier(self, model_path, image_path, image_name, labels = ["no_vest","vest"]):
        vest_classifier_model = torch.jit.load(model_path).eval().float().cuda()
        
        image = cv2.imread(f"{image_path}")
        resized_image = cv2.resize(image, (224, 224))
        resized_image = resized_image[:, :, ::-1].transpose(2, 0, 1)
        resized_image = np.ascontiguousarray(resized_image)
        resized_image = (
            torch.unsqueeze(torch.from_numpy(resized_image), 0).float().cuda()
        )
        resized_image /= 255.0
        transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        normalized_image = transform(resized_image)

        self.prediction = (
            torch.softmax(vest_classifier_model(normalized_image), dim=1)
            .cpu()
            .detach()
            .numpy()
        )
        
        
        print("Run {self.model} classfier on {image_name}")
    
    def run_pose_classifier(self, features, image_name, model_path):

        with open(model_path, "rb") as f:
            # TODO : (Nasha) Use onnx to import models
            # trunk-ignore(bandit/B301)
            self._pose_classifier = pickle.load(f)
        features = features.split(",")
        map_object = map(float, features)

        feat = np.asarray(list(map_object)).reshape((1, -1))
        # pose_classes are [0: "bad reach", 1: "random pose"]
        self.prediction = int(self._pose_classifier.predict(feat))
        
        

    def upload_predictions(self, conf = 0.5, image_name = "", labels = ["no_vest","vest"], tax_name="",image_classifier=False):
        if image_classifier:
            label = labels[1] if self.prediction[:, 1] > conf else labels[0]

            category_pred = CategoryPrediction(
            label=label,
            taxonomy_name=tax_name,
            reference_id=image_name,
            confidence=conf,
            class_pdf={labels[0]: round(float(self.prediction[:, 0]),2), labels[1]: round(float(self.prediction[:, 1]),2)}
            )
        else:
            # pose_classes are [0: "bad reach", 1: "random pose"]
            label = labels[self.prediction]

            category_pred = CategoryPrediction(
            label=label,
            taxonomy_name=tax_name,
            reference_id=image_name,
            ) 
        print(category_pred)

        dataset = self.dataset

        job = dataset.upload_predictions(
        model=self.model,
        predictions=[category_pred],
        update=True,
        asynchronous=False
        )


if __name__ == "__main__":
    #example with pose dataset
    ReachDataset= ScaleDataset()
    #get/create the dataset
    ReachDataset.get_dataset(dataset_id ="ds_c6wza8shv7h00ehtanjg")
    # #upload images in voxel dataset to scale
    source_dir = "/home/nasha_voxelsafety_com/voxel/experimental/nasha/data/activity_classifier/good_reach/"
    pose_embed = "5.286548803640718,-39.64911602730539,-19.824558013652695,6.608186004550902,14.538009210011978,21.146195214562873,-18.50292081274251,-7.929823205461076,-46.25730203185629,-21.146195214562873,1.32163720091018,30.397648179283223,2.643274401820359,30.397648179283223,3.964911602730538,30.397655620934128,3.964911602730539,26.432744018203596,-38.32747882639521,-1.3216372009101747,-31.719292821844313,0.0,5.286548803640718,60.795303800217354,6.608186004550898,56.830392197486816,-31.719292821844306,-40.970753228215564,-27.754381219113775,-39.64911602730539,-1.32163720091018,100.44441982752275,2.643274401820359,96.4795082247922,-31.719292821844306,-40.970753228215564,-27.754381219113775,-39.64911602730539,25.111106817293415,13.216372009101796,-5.286548803640719,-1.3216372009101818,-2.6432744018203636,0.0,-5.286548803640718,-5.286548803640713,13.216372009101795,-9.912275286000893,10.573097607281438,-8.590638085090715,-9.292574157126557,-8.139736024233843,-3.1441048034934487,11.196506550218341,-11.965062551634606,-11.947595302726308"
    ReachDataset.upload_to_dataset(image_name = "nasha_pose_classification_dataset_dani_good_reach_frame_103_101.jpg", pose_embedding = pose_embed)
    # upload ground truth labels to scale
    ReachDataset.add_gt_category(image_name= "nasha_pose_classification_dataset_dani_good_reach_frame_103_101.jpg", label ="good_reach", tax_name= "reach_clean")
    # create model
    ReachDataset.add_model(model_name ="Test Reach Classifier", model_ref = "reach-classifier-test", metadata= {"date":"01282022"})
    # run model on images
    ReachDataset.run_pose_classifier(features=pose_embed, image_name="nasha_pose_classification_dataset_dani_good_reach_frame_103_101.jpg", model_path="/home/nasha_voxelsafety_com/voxel/experimental/nasha/models/reach_classifer_01282022.sav")
    # upload predictions to scale
    ReachDataset.upload_predictions(conf = 0.5, image_path = source_dir, image_name = "nasha_pose_classification_dataset_dani_good_reach_frame_103_101.jpg", labels = ["bad_reach","random"], tax_name="reach_clean")


            
