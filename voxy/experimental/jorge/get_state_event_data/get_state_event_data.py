import base64

import boto3
from google.protobuf.any_pb2 import Any
from loguru import logger

# trunk-ignore-all(pylint/E0611)
from core.structs.protobufs.v1.event_pb2 import Event as EventPb
from core.structs.protobufs.v1.state_pb2 import State as StatePb


# trunk-ignore-all(pylint/C0116)
def main(*args):
    client = boto3.client("s3")

    response = client.get_object(
        Bucket="jorge-voxel-lambda-function-testing",
        Key="testdata",
    )

    for line in response["Body"].iter_lines():
        anypb = Any.FromString(base64.b64decode(line))

        if anypb.Is(EventPb.DESCRIPTOR):
            eventpb = EventPb()
            anypb.Unpack(eventpb)
            logger.info(eventpb)
        elif anypb.Is(StatePb.DESCRIPTOR):
            statepb = StatePb()
            anypb.Unpack(statepb)
            logger.info(statepb)
        else:
            logger.warning(f"unsupported message: {anypb}")


if __name__ == "__main__":
    main()
