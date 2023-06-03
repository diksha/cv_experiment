#!/usr/bin/env bash

set -eu

TMPDIR=$(mktemp -d)
trap 'rm -rf -- ${TMPDIR}' EXIT

echo "${PROVISIONER_PASSWORD_BASE64ENCODED}" | base64 -d > $TMPDIR/password.txt
("${STEP}" crypto jwk create --password-file "${TMPDIR}/password.txt" "${TMPDIR}/pub.json" "${TMPDIR}/priv.json" 2>/dev/null)

PRIV=$(cat "${TMPDIR}/priv.json" | "${STEP}" crypto jose format)

"${JQ}" -c --arg priv "$PRIV" --arg name "$PROVISIONER_NAME" '{
    "type": "JWK",
    "name": $name,
    "key": .,
    "encryptedKey": $priv,
    "options": {"x509": {}, "ssh": {}}
}' "${TMPDIR}/pub.json" | tr -d '\n' | jq -R -s '{"content": .}'