import json
import os

import django
from aws_lambda_powertools.utilities import parameters
from aws_lambda_powertools.utilities.parser import event_parser
from aws_lambda_powertools.utilities.parser.models import S3Model
from aws_lambda_powertools.utilities.typing import LambdaContext
from loguru import logger

# do our django setup
os.environ.setdefault("ENVIRONMENT", "development")
if os.environ["ENVIRONMENT"] != "development":
    for k, v in json.loads(parameters.get_secret("portal")).items():
        os.environ.setdefault(k, v)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core.portal.voxel.settings")
django.setup()

# no direct or indirect voxel model imports can happen before django setup
# trunk-ignore(pylint/C0413,flake8/E402): model imports have to come after setup
from core.portal.perceived_data.perceived_actor_state_duration_batch_processor import (
    PerceivedActorStateDurationBatchProcessor,
)


@event_parser(model=S3Model)
def lambda_handler(event: S3Model, context: LambdaContext):
    """Processes a single s3 event object

    Args:
        event (S3Model): S3 event object
        context (LambdaContext): lambda context metadata
    """
    bucket_name = event.Records[0].s3.bucket.name
    object_key = event.Records[0].s3.object.key
    if PerceivedActorStateDurationBatchProcessor.has_batch_been_processed(
        batch_key=object_key
    ):
        logger.error(f"This object has already been processed: {bucket_name}")
        return
    batch_processor = PerceivedActorStateDurationBatchProcessor()
    batch_processor.execute(bucket_name=bucket_name, batch_key=object_key)


if __name__ == "__main__":
    logger.info("Executing lambda handler with test params")
    lambda_handler(
        """
{
  "Records": [
    {
      "eventVersion": "2.0",
      "eventSource": "aws:s3",
      "awsRegion": "us-east-1",
      "eventTime": "1970-01-01T00:00:00.000Z",
      "eventName": "ObjectCreated:Put",
      "userIdentity": {
        "principalId": "EXAMPLE"
      },
      "requestParameters": {
        "sourceIPAddress": "127.0.0.1"
      },
      "responseElements": {
        "x-amz-request-id": "EXAMPLE123456789",
        "x-amz-id-2": "EXAMPLE123/5678abcdefghijklambdaisawesome/mnopqrstuvwxyzABCDEFGH"
      },
      "s3": {
        "s3SchemaVersion": "1.0",
        "configurationId": "testConfigRule",
        "bucket": {
          "name": "voxel-perception-staging-states-events",
          "ownerIdentity": {
            "principalId": "EXAMPLE"
          },
          "arn": "arn:aws:s3:::example-bucket"
        },
        "object": {
          "key": "test%2Fkey",
          "size": 1024,
          "eTag": "0123456789abcdef0123456789abcdef",
          "sequencer": "0A1B2C3D4E5F678901"
        }
      }
    }
  ]
}
""",
        None,
    )
