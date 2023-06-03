#!/bin/sh

set -eu

JQ_VERSION="1.6"
JQ_URL="https://github.com/stedolan/jq/releases/download/jq-${JQ_VERSION}/jq-linux64"
JQ_SHA256="af986793a515d500ab2d35f8d2aecd656e764504b789b66d7e1a0b727a124c44"

CACHE_DIR="/tmp/jq-${JQ_VERSION}-bin"
JQ="${CACHE_DIR}/jq"

if [ ! -x "${JQ}" ]; then
    mkdir -p "${CACHE_DIR}"
    curl -sSL "${JQ_URL}" -o "${JQ}"
    echo "${JQ_SHA256}  ${JQ}" | sha256sum -c 2>/dev/null 1>/dev/null
    chmod +x "${JQ}"
fi

exec "${JQ}" "$@"
