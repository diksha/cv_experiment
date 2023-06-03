# Voxel Edge Transcoder service

This service handles transcoding camera inputs and publishing them to Kinesis Video for consumption by the Perception system. the service runs as a greengrass component on edge devices which are distributed to customers. What follows is some basic information on how the service is laid out here, as well as how to perform some basic tasks related to this edge service.

## Provisioning

Provisioning new edges with this service requires knowing the following information ahead of time:

- The UUID of the edge (used as the IOT Thing Name)
- At least one camera rtsp uri
- The names of corresponding kinesis video streams provisioned for those cameras

### 1. Create an appropriate edge config in the production secrets manager

Create a new secret in aws secrets manager for this edge, named something like:

`edge/<customer>/<location>/edgeconfig.yaml`

The precise path does not matter as long as the configuration ends with `edgeconfig.yaml`. Also ensure that the location is tagged with a tag name of `edge:allow-uuid:primary` and a value of the UUID assigned to the edge.

The configuraiton file should be a yaml file that looks like the following:

```yaml
streams:
  - rtsp_url: <some-rtsp-uri-1>
    kinesis_video_stream: <some-kinesis-video-stream-1>
  - rtsp_url: <some-rtsp-uri-2>
    kinesis_video_stream: <some-kinesis-video-stream-2>
```

### 1. Deploy the appropriate greengrass component to your target device using the Greengrass UI

This can be achieved by either creating a new deployment specific to this edge, or adding the edge to an existing deployment.

## Development

Various development tasks can be completed mostly via bazel. Please reach out to jorge@voxelai.com if you are unsure how to complete any of these steps as the documentation here is not yet complete.

### Creating a new version

If any changes have been made to the `//services/edge/transcoder/container` targets, which include any changes to `//services/edge/transcoder/cmd/transcoder` you will need to create a release of the appropriate component. There are currently four components with associated bazel targets:

- `voxel.edge.QuicksyncTranscoder` -- `//services/edge/transcoder:quicksync.release`
- `voxel.edge.CudaTranscoder` -- `//services/edge/transcoder:cuda.release`
- `voxel.edge.QuicksyncTranscoderDev` -- `//services/edge/transcoder:quicksync-dev.release`
- `voxel.edge.CudaTranscoderDev` -- `//services/edge/transcoder:cuda-dev.release`

The components ending in `Dev` can be updated/released without causing any problems for production. Cutting a release currently takes the following steps:

1. Bump the version of the component by incrementing the `version` property of the appropriate `transcoder_release` rule in `services/edge/transcoder/BUILD.bazel`
2. Build the release with `bazel build //services/edge/transcoder:quicksync-dev.release`
3. Switch to a profile with enough production permissions to actually upload the release (currently only the AdministratorAccess profile for Production has enough access)
4. Perform the release with `bazel run //services/edge/transcoder:quicksync-dev.release`

To configure a profile for access in step 3, you can add the following lines to `$HOME/.aws/config`

```ini
[profile production-admin]
sso_start_url = https://voxelai.awsapps.com/start#/
sso_region = us-west-2
sso_account_id = 360054435465
sso_role_name = AdministratorAccess
region = us-west-2
```

Once these are in place, set the `AWS_PROFILE` environment variable to `production-admin` and run `aws sso login`

`export AWS_PROFILE=production-admin`
`./tools/aws sso login`

### Deploying a new version

Currently this process depends entirely on performing this manually through the AWS console.

1. Find the appropriate deployment listed under the AWS Greengrass console deployments list
2. Revise the deployment using the button in the top right
3. At the configure step, select the component you'd like to upgrade and click configure
4. Set the version number using the drop down at the top
5. Trigger the deployment
