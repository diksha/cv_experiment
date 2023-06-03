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
"""
We use custom loaders so we can bundle the index.html template with hashed
asset URLs (main.abcd1234.js), deploy to S3 or a local dev server, and
fetch index.html at runtime. Without this, we would need to include the
template in our web app container and would need to redeploy containers
when the file hashes change.

In the future we may be able to avoid this by maintaining a
directory of static asset builds, each nested within a hashed folder name,
while each filename has a consistent name. We couuld maintain a database
of build hashes and inject the desired build hash at runtime instead of
fetching the entire template with embedded hashes. This still gives us the
benefits of hashed filenames while also allowing us to maintain a history
of previous builds.

builds/
    hash1abc/
        scripts.js
        styles.css
    hash2xyz/
        scripts.js
        styles.css

Then at runtime we inject the hash directory via templating like:

    <script src="builds/{{ hash }}/scripts.js" />

See the Django docs for more info about custom loaders:
https://docs.djangoproject.com/en/3.2/ref/templates/api/#custom-loaders
"""
from typing import List

import boto3
from django.conf import settings
from django.template.base import Origin
from django.template.loader import TemplateDoesNotExist
from django.template.loaders.base import Loader


class S3StorageLoader(Loader):
    """Loads templates from S3 bucket."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not settings.AWS_STORAGE_BUCKET_NAME:
            raise RuntimeError(
                "The AWS_STORAGE_BUCKET_NAME setting is required when "
                "using S3StorageLoader."
            )
        s3 = boto3.resource("s3")
        self._bucket = s3.Bucket(settings.AWS_STORAGE_BUCKET_NAME)

    def get_template_sources(self, template_name: str) -> List[Origin]:
        matches = []
        for obj in self._bucket.objects.all():
            if obj.key.endswith(template_name):
                matches.append(
                    Origin(
                        name=obj.key,
                        template_name=template_name,
                        loader=self,
                    )
                )

        if not matches:
            raise TemplateDoesNotExist(
                "No matching templates found for "
                f"template_name: {template_name}"
            )
        return matches

    def get_contents(self, origin: Origin) -> str:
        obj = self._bucket.Object(origin.name)
        return obj.get()["Body"].read()
