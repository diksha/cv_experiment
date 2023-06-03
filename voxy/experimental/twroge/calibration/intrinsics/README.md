# Intrinsics Calibration Prediction

The intrinsics calibration prediction tool takes in a image directory and outputs a results file with the
model output for the universal spherical model and focal length.

```
./bazel run //experimental/twroge/calibration/intrinsics:predict_intrinsics -- --image_directory <image_dir> --output <output_directory>
```

These are the full arguments that need to get passed in:

```
usage: predict_intrinsics.py [-h] --image_directory IMAGE_DIRECTORY --output
                             OUTPUT

Predict Intrinsics

optional arguments:
  -h, --help            show this help message and exit
  --image_directory IMAGE_DIRECTORY
                        The image directory for
  --output OUTPUT       The output directory to put the model
```
