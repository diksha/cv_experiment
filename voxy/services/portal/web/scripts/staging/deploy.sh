#!/bin/bash
##
## Copyright 2020-2022 Voxel Labs, Inc.
## All rights reserved.
##
## This document may not be reproduced, republished, distributed, transmitted,
## displayed, broadcast or otherwise exploited in any manner without the express
## prior written permission of Voxel Labs, Inc. The receipt or possession of this
## document does not convey any rights to reproduce, disclose, or distribute its
## contents, or to manufacture, use, or sell anything that it may describe, in
## whole or in part.
##

set -euo pipefail

# Deploy Django app frontends
ENVIRONMENT=staging \
	DJANGO_SECRET_KEY=__unused__ \
	./bazel run //core/portal:deploy -- --no-input

# Deploy SPA frontends
./services/portal/web/scripts/build.sh
./tools/aws s3 sync \
	./services/portal/web/build \
	s3://voxel-portal-staging-static/static/frontend

# Invalidate the CloudFront cache
AWS_PAGER="" \
	./tools/aws cloudfront create-invalidation --distribution-id E1EGPUW4HF020Z --paths "/*"
