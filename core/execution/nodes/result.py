import torch

# Model
# def get_predictions(img):
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
results = model('/home/diksha_voxelsafety_com/sample_proj/cv_experiment/data/sample_imgs/sample_1.jpeg')
results.print()