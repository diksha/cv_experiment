# Fragment Archival

Persisted video from kinesis is stored in an S3 bucket in the form of several short video files called fragments which are a maximum of 10 seconds long.

The format of the object keys for these fragments in S3 is described [here](./key/README.md).

## Fragment Metadata

Along with the video data, we store custom metadata on the fragments. These values are:

`x-amz-meta-camera-uuid`: UUID of the camera from which the fragment originated.

`x-amz-meta-duration-ms`: duration of the video fragment in milliseconds.

`x-amz-meta-fragment-number`: the unique identifier of the fragment in Kinesis Video Archived Media stream.

`x-amz-meta-producer-timestamp-ms`: the timestamp from the producer corresponding to the fragment. Unix timestamp in milliseconds.

`x-amz-meta-server-timestamp-ms`: the timestamp from the AWS server corresponding to the fragment. Unix timestamp in milliseconds.
