#!/bin/bash

# Start minikube cluster if not already running
if ./tools/minikube status | grep -q "host: Running"; then
	echo "✅ minikube is already running"
else
	./tools/minikube start --driver none
fi

PATH=$PWD/tools:$PATH \
	./tools/skaffold dev --kube-context minikube -f skaffold.yaml
