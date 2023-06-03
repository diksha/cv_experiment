#!/bin/bash
##
## Copyright 2020-2022 Voxel Labs, Inc.
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

while getopts c:o:t: option; do
	case "${option}" in

	c) CAMERA_LIST_FILE=${OPTARG} ;;
	o) OUTPUT_DIR=${OPTARG} ;;
	*) exit ;;
	esac
done

if [ -z "$CAMERA_LIST_FILE" ]; then
	echo "exit: No CAMERA_LIST_FILE specified"
	exit
fi

if [ -z "$OUTPUT_DIR" ]; then
	echo "exit: No OUTPUT_DIR specified"
	exit
fi

# Make sure the output directory exists
mkdir -p "$OUTPUT_DIR"

while IFS= read -r rtsp_uri; do
	# Construct a filename without auth values or illegal filename characters
	without_scheme_and_auth=$(echo "$rtsp_uri" | sed -e 's/rtsp:\/\///g' | rev | cut -d'@' -f 1 | rev)
	sanitized_filename=$(echo "$without_scheme_and_auth" | sed -e 's/[^A-Za-z0-9._-]/_/g') # trunk-ignore(shellcheck/SC2001)

	# Run ffmpeg for the desired amount of time
	echo "Capturing clip for: $rtsp_uri"
	ffmpeg -nostdin -y -rtsp_transport tcp -i "$rtsp_uri" -an -ss 10 -q:v 1 -frames:v 1 "$OUTPUT_DIR/$sanitized_filename.jpg"
done < <(grep "" "$CAMERA_LIST_FILE")

echo "Done"
