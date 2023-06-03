#
# Copyright 2020-2021 Voxel Labs, Inc.
# All rights reserved.
#
# This document may not be reproduced, republished, distributed, transmitted,
# displayed, broadcast or otherwise exploited in any manner without the express
# prior written permission of Voxel Labs, Inc. The receipt or possession of this
# document does not convey any rights to reproduce, disclose, or distribute its
# contents, or to manufacture, use, or sell anything that it may describe, in
# whole or in part.
#
import boto3
from botocore import UNSIGNED
from botocore.client import Config


def get_boto_client_with_no_creds(aws_service: str) -> boto3.client:
    """Returns a boto3 client that does not require access to the network or to credentals.
    This is important since most tests should not be able to change our real AWS data.

    :type aws_service: str
    :param aws_service: s3, sts, etc
    :returns: A boto3 client that does not require access to the network or to credentals.
    :rtype: boto3.client
    """

    return boto3.client(aws_service, config=Config(signature_version=UNSIGNED))
