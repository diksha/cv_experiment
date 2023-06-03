#!/bin/bash

# API service
./core/portal/scripts/_deploy_eks_service.sh \
	-b portal \
	-e staging \
	-i "$BUILDKITE_COMMIT"
