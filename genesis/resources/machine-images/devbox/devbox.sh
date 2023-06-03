#!/bin/bash
# This script should only be run as root user. It is recommended that the userdata/entrypoint call this script
set -euo pipefail

_setup_user_() {
	echo "Setting up User!"
	groupadd --gid "${DEVBOX_GROUP_ID}" "${DEVBOX_GROUP}"
	useradd --shell /usr/bin/bash --gid "${DEVBOX_GROUP_ID}" --uid "${DEVBOX_USER_ID}" --home-dir "${DEVBOX_USER_HOME}" --create-home "${DEVBOX_USER}"
	mkdir -p "${DEVBOX_USER_HOME}/.ssh"
	echo "${DEVBOX_USER_PUBLIC_KEY_BASE_64}" | base64 -d >"${DEVBOX_USER_HOME}/.ssh/authorized_keys"
	chown -R "${DEVBOX_USER}:${DEVBOX_GROUP}" "${DEVBOX_USER_HOME}/.ssh"
	chmod 700 "${DEVBOX_USER_HOME}/"
	chmod 700 "${DEVBOX_USER_HOME}/.ssh"
	chmod 600 "${DEVBOX_USER_HOME}/.ssh/authorized_keys"
	usermod -aG sudo "${DEVBOX_USER}"
	usermod -aG docker "${DEVBOX_USER}"
	(echo /etc/sudoers | grep "${DEVBOX_USER}" &>/dev/null) || echo "${DEVBOX_USER} ALL=(ALL:ALL) NOPASSWD:ALL" >>/etc/sudoers
}

start() {
	# export DEVBOX_USER=sid
	# export DEVBOX_USER_ID=1111
	# export DEVBOX_GROUP=voxel
	# export DEVBOX_GROUP_ID=2222
	# export DEVBOX_USER_PUBLIC_KEY_BASE_64=c3NoLXJzYSBBQUFBQjNOemFDMXljMkVBQUFBREFRQUJBQUFCZ1FDNGJhVzJxNmtMZFR1ZDAxRlRKcmt3aFlFcDFFM0tWeW5QRlF6dGdTc2g3c0RjNjFHYmtTZmxKd1hKOUJ4ekVrdDhIaDIyR3o1RnJENFBvOVBNZGYzcllYbHovY2xnVExFcUNjQkFxUkN2WGpMdWhjczlBQzI2YlRFaWZzVEh0aWg0SzZocldzQks4eHNiZ2MvMnRNcEhJRk01UUNONDhtT0dzaXRMQjNSR1poVWpEdXM5RGNFdEtUM0p4SGdHcVZZQ0oyd01zcGtnTGN6N1labGFobGRuQyt2bjFlNC9wWmh1T3lDL2FtcTlpd1BCUG4vRXVrWXEvekkwRUlJUDhydTNReG5TMWx3MmljczMxM0U0YTJuL1lTM2VyME9Cd0toNVI4bUhBWXhUcStzbyt6Vng4Z3JnQnF2SlBnaU9tUW41Q0xuYkg4WE5Nbk5XaElpY2NibUQ0NzBSRkp1QmMrZ1ZDR1VodldjVVFqc2JaRkRteFNUaTZUQ3hZQ2x1TGpVQWtxc0t5MkdyY0dyN0Fqb2RkMmpyM1Q5TmtIN1cySHQ2M1dwdnRkMVhOMWNHUWp4M3lQcGFUTnpCMlg0SFlKdWthS2Y1MlE0QzhBOGdkZ2JjUzFEZHZ0QjJLdDJCRUFKdVdtekl0aHo0M096UGYybExBMHFZWm41Mk9CQVdIcjg9IHN5c3RlbXVzZXJAMTkyLjE2OC4xLjIK

	DEVBOX_USER="${DEVBOX_USER:?'DEVBOX_USER needs to be defined'}"
	DEVBOX_GROUP="${DEVBOX_GROUP:?'DEVBOX_GROUP needs to be defined'}"
	DEVBOX_USER_ID="${DEVBOX_USER_ID:?'DEVBOX_USER_ID needs to be defined'}"
	DEVBOX_GROUP_ID="${DEVBOX_GROUP_ID:?'DEVBOX_GROUP_ID needs to be defined'}"
	DEVBOX_USER_PUBLIC_KEY_BASE_64="${DEVBOX_USER_PUBLIC_KEY_BASE_64:?'DEVBOX_USER_PUBLIC_KEY_BASE_64 needs to be defined'}"
	DEVBOX_USER_HOME="/home/${DEVBOX_USER}"

	id "${DEVBOX_USER}" &>/dev/null || _setup_user_
}

"${1:-start}" "${@:2}"
