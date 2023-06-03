# Yolo downsampling

Copy images from voxel-datasets/.... to voxel-lightly-input.

Run lightly downsample script using lightly

Get list of downsampled images from docker container

```shell
cat filenames/sampled_filenames_excluding_datapool.txt
```

Do train_val split using split_train_val.py

```shell
./bazel run experimental/diksha/yolo:split_train_val
```

Upload training.txt, validation.txt, yolo_training_dataset.yaml to s3 location /training folder.
[Example training folder](https://s3.console.aws.amazon.com/s3/buckets/voxel-users?region=us-west-2&prefix=diksha/yolo/training/)

Run yolo training buildkite pipeline
