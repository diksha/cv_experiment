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

# Generate migration files
ENVIRONMENT=development \
	./bazel run //core/portal:manage -- makemigrations

# Sync Bazel output to workspace source code
FILES=$(find bazel-out/k8-fastbuild/bin/core/portal/manage.runfiles/voxel/core/portal/**/migrations/*.py)

while IFS= read -r FILE; do
	# Files which were not created/modified during the build phase
	# are symlinks here, so we're interested in non-symlink files
	if ! [[ -L $FILE ]]; then
		# This is a new migration file, copy it to workspace
		prefix_to_remove="bazel-out/k8-fastbuild/bin/core/portal/manage.runfiles/voxel/"
		destination_file=${FILE/#$prefix_to_remove/}
		cp "$FILE" "$destination_file"
		echo "Copying migration file: $destination_file"
	fi
done <<<"$FILES"
