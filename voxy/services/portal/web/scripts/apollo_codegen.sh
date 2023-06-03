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

echo "Removing existing gencode..."
rm services/portal/web/src/__generated__/*

./tools/apollo codegen:generate \
	--config=services/portal/web/apollo.config.js \
	--localSchemaFile=core/portal/lib/graphql/schema.graphql \
	--target=typescript \
	--outputFlat=services/portal/web/src/__generated__/
