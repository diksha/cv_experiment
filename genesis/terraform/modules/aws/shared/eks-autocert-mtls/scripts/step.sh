#!/bin/sh

set -eu

STEP_VERSION="0.23.2"
STEP_FILENAME="step_linux_${STEP_VERSION}_amd64.tar.gz"
STEP_URL="https://github.com/smallstep/cli/releases/download/v${STEP_VERSION}/${STEP_FILENAME}"
STEP_SHA256=044fe5517ece907dd9193e92ba1579926310e0694e83fe947b428e24ff089785

CACHE_DIR=/tmp/step-cli-cache
STEP="${CACHE_DIR}/step_${STEP_VERSION}/bin/step"

if [ ! -x "${STEP}" ]; then
    mkdir -p "${CACHE_DIR}"
    curl -sSL "${STEP_URL}" -o "${CACHE_DIR}/${STEP_FILENAME}"
    echo "${STEP_SHA256}  ${CACHE_DIR}/${STEP_FILENAME}" | sha256sum -c 1>/dev/null 2>/dev/null
    (cd "${CACHE_DIR}" && tar xzf "${STEP_FILENAME}")
fi

exec "${STEP}" "$@"
