from typing import Iterable, Tuple

from pyflink.datastream import FlatMapFunction, RuntimeContext

from protos.platform.bowser.v1.bowser_config_consumer_pb2 import (
    ProcessorConsumerAwsS3Bucket,
)


# trunk-ignore-all(pylint/E0611): ignore pb import errors
class AwsBucketToS3ObjectFlatMap(FlatMapFunction):
    def __init__(self, client):
        self._aws_client = client

    def open(self, runtime_context: RuntimeContext) -> None:
        """Initializes the boto3 client

        :param  RuntimeContext runtime_context: bowser runtime context
        """
        self._aws_client = self._aws_client.client("s3")

    def flat_map(
        self, bucket: ProcessorConsumerAwsS3Bucket
    ) -> Iterable[Tuple[str, str]]:
        """Receives bucket config and produce a Tuple of buclet and folder uris
        Args:
            bucket (Tuple[str, str]): tuple of s3 objects

        Yields:
            state or event or null messages
        """

        for location in bucket.uris:
            response = self._aws_client.list_objects_v2(
                Bucket=bucket.name,
                Prefix=location,
            )

            keys = [o["Key"] for o in response["Contents"]]

            while response["IsTruncated"]:
                response = self._aws_client.list_objects_v2(
                    Bucket=bucket.name,
                    Prefix=location,
                    ContinuationToken=response["NextContinuationToken"],
                )

                keys.extend([o["Key"] for o in response["Contents"]])

            keys.sort(
                key=lambda x: "-".join(x.split("/")[-1].split("-")[4:-5])
            )

            for k in keys:
                yield (bucket.name, k)
