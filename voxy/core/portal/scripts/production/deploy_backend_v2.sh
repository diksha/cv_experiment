#!/bin/bash

# API service
./core/portal/scripts/_deploy_eks_service.sh \
	-b portal \
	-e production \
	-i "$BUILDKITE_COMMIT"
