import torchvision
import os


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, root_directory, annotation_file, feature_extractor):
        super(CocoDetection, self).__init__(root_directory, annotation_file)
        self.feature_extractor = feature_extractor

    def __getitem__(self, index):
        # read in PIL image and target in COCO format
        image, target = super(CocoDetection, self).__getitem__(index)
        # preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
        image_id = self.ids[index]
        target = {'image_id': image_id, 'annotations': target}
        target['annotations'] = [item for item in target['annotations'] if item['category_id'] > 0 ] #remove negative annotations
        encoding = self.feature_extractor(images=image, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze() # remove batch dimension
        target = encoding["labels"][0] # remove batch dimension
        return pixel_values, target
