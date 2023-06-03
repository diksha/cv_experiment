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

# ============================================================================
# Copy and run this script on an edge server to audit it's configuration
# ============================================================================

# ============================================================================
# Required programs
# ============================================================================

check_program_installed() {
	local program_name=$1
	local expected_output=$2
	local output="$(which "$1")"
	if [[ $output == *"$program_name"* ]]; then
		echo "  ✅ $program_name is installed"
	else
		echo "  ❌ $program_name is NOT installed"
	fi
}

echo "Required programs:"
check_program_installed "nvidia-smi"
check_program_installed "java"
check_program_installed "docker"
check_program_installed "docker-compose"

# ============================================================================
# Required services
# ============================================================================

check_service_state() {
	local service_name=$1
	local state="$(systemctl show --property ActiveState "$service_name")"
	if [[ $state == *"ActiveState=active"* ]]; then
		echo "  ✅ $service_name is running"
	else
		echo "  ❌ $service_name is NOT running"
	fi
}

echo
echo "Required services:"
check_service_state "docker"
check_service_state "greengrass.service"
check_service_state "ssh.service"

# ============================================================================
# Required configuration
# ============================================================================

echo
echo "Required configuration:"

# Password based SSH login is disabled
ssh_password_disabled="$(cat /etc/ssh/sshd_config | grep -E ^PasswordAuthentication\ no\$)"
if [[ -n $ssh_password_disabled ]]; then
	echo "  ✅ ssh password authentication is disabled"
else
	echo "  ❌ ssh password authentication is NOT disabled"
fi

# TODO: Greengrass starts on system startup
# TODO: Admin user created
# TODO: SSH public key present

# ============================================================================
# Optional configuration
# ============================================================================

echo
echo "Optional configuration:"

check_service_state "rtsp-server"
check_service_state "edvrserver"
