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

set -e

# Usage:
# core/infra/edge/open_tunnel.sh -t <aws-iot-thing-name>

while getopts t: option; do
	case "${option}" in

	t) AWS_IOT_THING_NAME=${OPTARG} ;;
	*) exit ;;
	esac
done

if [ -z "$AWS_IOT_THING_NAME" ]; then
	echo "exit: No AWS_IOT_THING_NAME specified"
	exit
fi

openTunnelResponse=$(./tools/aws iotsecuretunneling open-tunnel --output yaml --destination-config thingName="$AWS_IOT_THING_NAME",services=SSH)
tunnelId=$(echo "$openTunnelResponse" | grep tunnelId | awk '{print $2}')
sourceAccessToken=$(echo "$openTunnelResponse" | grep sourceAccessToken | awk '{print $2}')

exit_script() {
	# Close the tunnel when we're done
	echo "Cleaning up..."
	./tools/aws iotsecuretunneling close-tunnel --tunnel-id "$tunnelId"
	# Clear the trap
	trap - SIGINT SIGTERM
	# Send SIGTERM to child/sub processes
	kill -- -$$
}
trap exit_script SIGINT SIGTERM

# Open the tunnel
./tools/aws_iot_securetunneling_localproxy -r us-west-2 -s 5555 -t "$sourceAccessToken"
