#!/bin/bash

set -euo pipefail

# Deploy Django app frontends
ENVIRONMENT=staging \
	DJANGO_SECRET_KEY=__unused__ \
	DEPLOYMENT_PROVIDER=eks-final \
	./bazel run //core/portal:deploy -- --no-input

# Deploy SPA frontends
./services/portal/web/scripts/build.sh
./tools/aws s3 sync \
	./services/portal/web/build \
	s3://voxel-portal-staging-static-resources/static/frontend

# Invalidate the CloudFront cache
AWS_PAGER="" \
	./tools/aws cloudfront create-invalidation --distribution-id E1B7C6A6O09V60 --paths "/*" --profile staging
