#!/bin/bash
##
## Copyright 2020-2021 Voxel Labs, Inc.
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

# Wait for database service to be available
wait_for_database() {
	until ./tools/kubectl get pods postgres-db -o jsonpath="{.status.phase}" | grep -q "Running"; do
		echo "Waiting for database..."
		sleep 5
	done
}
export -f wait_for_database # export required so function is available to subprocess
timeout --foreground 60s bash -c wait_for_database

# Run migrations
./core/portal/scripts/development/migrate.sh

# Start interactive dev server
AUTH0_TENANT_DOMAIN="voxeldev.us.auth0.com" \
	AUTH0_API_IDENTIFIER="61afbe5d7519a004cc5f3921" \
	AUTH0_JWT_AUDIENCE="http://localhost:9000" \
	AUTH0_MANAGEMENT_CLIENT_ID="o8wT4ahbl90T3ZbTkEnXKpR2Pa4gEFbk" \
	AUTH0_MANAGEMENT_CLIENT_SECRET="iqb1QjGQyhZSjcGeVohlznsLPXv4w2VzeceQCvbZMErzbuM0xBbXKwmqkC086BzD" \
	./ibazel run //core/portal:devserver
