# Running Production Graph Locally

There are two types of local runs you can do for the production graph - development and staging.

Development is easy to run and is going to be the one you want to use in most cases.

Staging is a bit more of a pain to run, and should only really be used when you require cloud infrastructure which is not available in development (i.e. something made in terraform)

Note: **Ensure you are logged into aws and have the correct profile set**

## Development

By default all publishing is disabled for the development configuration. Make sure to use the `default` AWS profile which provided
by the configuration file at `third_party/aws/config` in this repo. See the relevant [README.md](../../../third_party/aws/readme.md)

In order to enable publishing, change the indicated configuration values in ~/

The environment is set to development by default, so you just need to run:

```shell
bazel run //core/execution/runners:production -- --camera_config_path configs/cameras/wesco/reno/0002/cha.yaml --no-serialize_logs
```

### Publisher testing

Before running the production graph, open a separate terminal run the following to emulate Google Cloud Platform's Pub/Sub.

```shell
gcloud --quiet beta emulators pubsub start
```

## Staging

In the staging environment, we do not have access to production Kinesis Video Streams. For that reason, you must populate a video stream in the Staging account. To do this, use the `kvspusher` tool located at `voxel/services/edge/transcoder/cmd/kvspusher/main.go`. With this tool you can populate a video stream from either a default test source, RTSP stream, or another kinesis video stream.

Once you have populated a KVS stream in staging, provide the ARN of the stream in the [staging config](../../../configs/graphs/production/environment/staging.yaml).

From there, set your AWS profile to staging, and specify the `--environment staging` flag.

```shell
AWS_PROFILE=staging bazel run //core/execution/runners:production -- --environment staging --no-serialize_logs
```

## Notes

- Adding the `--no-serialize_logs` flag will produce more human-readable logs

## Troubleshooting

### Installing gcloud

Follow <https://cloud.google.com/sdk/docs/install-sdk>

Install emulator with the following commands

```shell
gcloud components install pubsub-emulator
gcloud components update
```

### Emulator port is busy

Either kill the process using the port or you can change the port in `configs/graphs/production/environment/development.yaml`

### Errors Assuming roles

If you get errors that look like any of

- `botocore.errorfactory.UnauthorizedException: An error occurred (UnauthorizedException) when calling the GetRoleCredentials operation: Session token not found or invalid`
- `botocore.exceptions.ClientError: An error occurred (AccessDenied) when calling the AssumeRole operation`

Ensure that you have authenticated with aws and have set your profile to **Prime > Developer Access**.
