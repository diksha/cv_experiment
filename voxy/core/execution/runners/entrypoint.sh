#!/usr/bin/env bash

set -euo pipefail

export ROOT_DIR="${ROOT_DIR:-"/app"}"

if [[ -n ${CONFIG_FILE_BASE64_ENCODED:-} ]]; then
	CONFIG_FILE="$(mktemp)"
	export CONFIG_FILE
	echo "${CONFIG_FILE_BASE64_ENCODED}" | base64 -d >"${CONFIG_FILE}"
fi

if [[ -z ${ENVIRONMENT:-} ]]; then
	echo "environment variable ENVIRONMENT must be defined to describe the execution environment (production)" 1>&2
	exit 1
fi

CONFIG_FILE="${CONFIG_FILE:?'CONFIG_FILE need to be defined as a file path or base64 encoded via env var CONFIG_FILE_BASE64_ENCODED'}"

if [[ -n ${GOOGLE_APPLICATION_CREDENTIALS_BASE64_ENCODED:-} ]]; then
	GOOGLE_APPLICATION_CREDENTIALS="$(mktemp)"
	export GOOGLE_APPLICATION_CREDENTIALS
	echo "${GOOGLE_APPLICATION_CREDENTIALS_BASE64_ENCODED}" | base64 -d >"${GOOGLE_APPLICATION_CREDENTIALS}"
fi

echo "Running in single mode. Camera config file at ${CONFIG_FILE}!"
# Send SIGKILL signal after 10 seconds of timeout completing if for some reason (like opentelemetry getting stuck) SIGTEM doesn't exits.
timeout -k 10 --preserve-status "${TIMEOUT:-43200}" /usr/bin/python3 "${ROOT_DIR}/core/execution/runners/production" --camera_config_path "${CONFIG_FILE}" --environment "${ENVIRONMENT}"
