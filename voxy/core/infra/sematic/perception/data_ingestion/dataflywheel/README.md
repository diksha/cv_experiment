# Dataflywheel Pipeline

We have created a process of getting data from our production environment where we do not do well, label them and feeding it to our machine learning models for our models to perform better. Various steps involved in dataflywheel are:

1. Get various incidents from portal.
2. Run lightly downsample using datapool feature. Datapool feature allows users to incrementally build up a dataset, keeps track of the representations of previously selected samples and uses this information to pick new samples in order to maximize the quality of the final dataset.
3. For videos that have not been previous ingested, ingest them to metverse and voxel-logs.

   **To take care of failed states of this pipeline, we only mark videos as ingested, once scale tasks are created for them.**

4. Send labeling tasks to scale

## Types of downsampling

### Sequence selection(short length videos)

Some pipelines can allow lightly to select sequences of frames from a video, which is determined by config value **selected_sequence_length**

### Image selection(list of downsampled images)

Other pipelines downsample input data to list of downsampled images. Example config **core/ml/data/curation/configs/PPE_DATAFLYWHEEL.yaml**

## Lightly version

We define the lightly version in lightly_worker.py which is used in local runs. We also create a sematic lightly docker container that is used by sematic cloud resolver. To update lightly image use the following buildkite pipeline.

[Update lightly buildkite pipeline](https://buildkite.com/voxel/update-lightly-image)
