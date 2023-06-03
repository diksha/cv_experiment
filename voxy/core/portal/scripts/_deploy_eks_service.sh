#!/bin/bash

set -eua

while getopts b:i:e:c:s:g: option; do
	case "${option}" in

	b) BASE_NAME=${OPTARG} ;;
	e) DEPLOY_ENV=${OPTARG} ;;
	i) IMG_TAG=${OPTARG} ;;
	*) exit ;;
	esac
done

if [ -z "$BASE_NAME" ]; then
	echo "exit: No BASE_NAME specified"
	exit
fi

if [ -z "$DEPLOY_ENV" ]; then
	echo "exit: No DEPLOY_ENV specified"
	exit
fi

if [ -z "$IMG_TAG" ]; then
	echo "exit: No IMG_TAG specified"
	exit
fi

SERVICE_NAME="${BASE_NAME}-${DEPLOY_ENV}-api"
# ----------------------------------------------------------------------------
# Database migration task
# ----------------------------------------------------------------------------
MIGRATION_APP_NAME="$SERVICE_NAME-migrate"
echo "-------------------------------------"
echo "Updating migration app..."
echo "-------------------------------------"
echo "Buildkite Commit:  $IMG_TAG"
echo
MIGRATION_APP_DEPLOY=$(./tools/argocd app set "$MIGRATION_APP_NAME" --helm-set image.tag="$IMG_TAG" --auth-token "$ARGO_AUTH_TOKEN" --grpc-web --server argo.voxelplatform.com)

echo "waiting for migration result... $MIGRATION_APP_DEPLOY"
MIGRATION_APP_DEPLOY_RESULT=$(./tools/argocd app wait "$MIGRATION_APP_NAME" --auth-token "$ARGO_AUTH_TOKEN" --grpc-web --server argo.voxelplatform.com)
if [ -n "$MIGRATION_APP_DEPLOY_RESULT" ]; then
	echo "Successfully updated migration"
	echo
else
	echo "ERROR: Failed to finish migration: $IMG_TAG"
	exit 1
fi

# ----------------------------------------------------------------------------
# API service
# ----------------------------------------------------------------------------
echo "-----------------------------------"
echo "Updating API service task definition..."
echo "-----------------------------------"
echo "Buildkite Commit:   $IMG_TAG"
echo

SERVICE_APP_DEPLOY=$(./tools/argocd app set "$SERVICE_NAME" --helm-set image.tag="$IMG_TAG" --auth-token "$ARGO_AUTH_TOKEN" --grpc-web --server argo.voxelplatform.com)

echo "waiting for deploy result... $SERVICE_APP_DEPLOY"
SERVICE_APP_DEPLOY_RESULT=$(./tools/argocd app wait "$SERVICE_NAME" --auth-token "$ARGO_AUTH_TOKEN" --grpc-web --server argo.voxelplatform.com)
if [ -n "$SERVICE_APP_DEPLOY_RESULT" ]; then
	echo "Successfully updated api update"
	echo
else
	echo "ERROR: Failed to finish api update: $IMG_TAG"
	exit 1
fi

# ----------------------------------------------------------------------------
# State message worker service
# ----------------------------------------------------------------------------
STATE_MESSAGE_SERVICE_NAME="$SERVICE_NAME-state-message-worker"
echo "-----------------------------------"
echo "Updating state message worker service task definition..."
echo "-----------------------------------"
echo "Buildkite Commit:   $IMG_TAG"
echo
STATE_MESSAGE_SERVICE_DEPLOY=$(./tools/argocd app set "$STATE_MESSAGE_SERVICE_NAME" --helm-set image.tag="$IMG_TAG" --auth-token "$ARGO_AUTH_TOKEN" --grpc-web --server argo.voxelplatform.com)

echo "waiting for state message worker results... $STATE_MESSAGE_SERVICE_DEPLOY"
STATE_MESSAGE_SERVICE_DEPLOY_RESULT=$(./tools/argocd app wait "$STATE_MESSAGE_SERVICE_NAME" --auth-token "$ARGO_AUTH_TOKEN" --grpc-web --server argo.voxelplatform.com)
if [ -n "$STATE_MESSAGE_SERVICE_DEPLOY_RESULT" ]; then
	echo "Successfully updated state message worker"
	echo
else
	echo "ERROR: Failed to finish state message worker update: $IMG_TAG"
	exit 1
fi

# ----------------------------------------------------------------------------
# Event message worker service
# ----------------------------------------------------------------------------
EVENT_MESSAGE_SERVICE_NAME="$SERVICE_NAME-event-message-worker"
echo "-----------------------------------"
echo "Updating event message worker service task definition..."
echo "-----------------------------------"
echo "Buildkite Commit:   $IMG_TAG"
echo

EVENT_MESSAGE_SERVICE_DEPLOY=$(./tools/argocd app set "$EVENT_MESSAGE_SERVICE_NAME" --helm-set image.tag="$IMG_TAG" --auth-token "$ARGO_AUTH_TOKEN" --grpc-web --server argo.voxelplatform.com)

echo "waiting for event message worker results... $EVENT_MESSAGE_SERVICE_DEPLOY"
EVENT_MESSAGE_SERVICE_DEPLOY_RESULT=$(./tools/argocd app wait "$EVENT_MESSAGE_SERVICE_NAME" --auth-token "$ARGO_AUTH_TOKEN" --grpc-web --server argo.voxelplatform.com)
if [ -n "$EVENT_MESSAGE_SERVICE_DEPLOY_RESULT" ]; then
	echo "Successfully updated event message worker"
	echo
else
	echo "ERROR: Failed to finish event message worker update: $IMG_TAG"
	exit 1
fi
