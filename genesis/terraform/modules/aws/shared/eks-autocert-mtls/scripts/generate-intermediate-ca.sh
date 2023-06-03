#!/usr/bin/env bash

set -eu

TMPDIR=$(mktemp -d)
trap 'rm -rf -- ${TMPDIR}' EXIT

# REQUIRED ENV ARGUMENTS
#
# ROOT_CA_CERT_PEM_BASE64ENCODED
# ROOT_CA_KEY_PEM_BASE64ENCODED
# ROOT_CA_PASSWORD_BASE64ENCODED
# INTERMEDIATE_CA_PASSWORD_BASE64ENCODED
# INTERMEDIATE_CA_COMMON_NAME

ROOT_CA_CERT="${TMPDIR}/root-ca.crt"
ROOT_CA_KEY="${TMPDIR}/root-ca.key"
ROOT_CA_PASSWORD_FILE="${TMPDIR}/root-ca.password.txt"

INTERMEDIATE_CA_CERT="${TMPDIR}/intermediate-ca.crt"
INTERMEDIATE_CA_KEY="${TMPDIR}/intermediate-ca.key"
INTERMEDIATE_CA_PASSWORD_FILE="${TMPDIR}/intermediate-ca.password.txt"

# this data is all pretty sensitive, we should definitely find a better way
# to do this eventually but for now we'll only be writing these files to a temp
# folder when these modules initially spin up, and the trap above should take care
# of wiping them when this program exits (even if from an error)
echo "${ROOT_CA_CERT_PEM_BASE64ENCODED}" | base64 -d > "${ROOT_CA_CERT}"
echo "${ROOT_CA_KEY_PEM_BASE64ENCODED}" | base64 -d > "${ROOT_CA_KEY}"
echo "${ROOT_CA_PASSWORD_BASE64ENCODED}" | base64 -d > "${ROOT_CA_PASSWORD_FILE}"
echo "${INTERMEDIATE_CA_PASSWORD_BASE64ENCODED}" | base64 -d > "${INTERMEDIATE_CA_PASSWORD_FILE}"

("${STEP}" certificate create "${INTERMEDIATE_CA_COMMON_NAME}" "${INTERMEDIATE_CA_CERT}" "${INTERMEDIATE_CA_KEY}" \
    --password-file "${INTERMEDIATE_CA_PASSWORD_FILE}" \
    --profile intermediate-ca \
    --ca "${ROOT_CA_CERT}" \
    --ca-key "${ROOT_CA_KEY}" \
    --ca-password-file "${ROOT_CA_PASSWORD_FILE}" 2>/dev/null)

INTERMEDIATE_CA_CERT_BASE64ENCODED=$(cat "${INTERMEDIATE_CA_CERT}" | base64 -w 0)
INTERMEDIATE_CA_KEY_BASE64ENCODED=$(cat "${INTERMEDIATE_CA_KEY}" | base64 -w 0)

"${JQ}" --null-input --arg cert "${INTERMEDIATE_CA_CERT_BASE64ENCODED}" --arg key "${INTERMEDIATE_CA_KEY_BASE64ENCODED}" '{"cert": $cert, "key": $key}'