#!/bin/bash

# Check if the service account exists
while true; do
	service_account_status=$(./tools/minikube kubectl -- get serviceaccounts default | grep -c default)
	if [[ $service_account_status -gt 0 ]]; then
		echo "Service account is ready"
		break
	else
		echo "Waiting for service account to be created..."
		sleep 5
	fi
done
