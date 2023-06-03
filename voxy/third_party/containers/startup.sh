#!/bin/bash
source "$(dirname "$(realpath "$0")")/logger.sh"

wait_for_process() {
	local max_time_wait=30
	local process_name="$1"
	local waited_sec=0
	while ! pgrep "$process_name" >/dev/null && ((waited_sec < max_time_wait)); do
		INFO "Process $process_name is not running yet. Retrying in 1 seconds"
		INFO "Waited $waited_sec seconds of $max_time_wait seconds"
		sleep 1
		((waited_sec = waited_sec + 1))
		if ((waited_sec >= max_time_wait)); then
			return 1
		fi
	done
	return 0
}

startup() {
	INFO "Starting supervisor"
	/usr/bin/supervisord -n >>/dev/null 2>&1 &

	INFO "Waiting for processes to be running"
	processes=(dockerd)

	for process in "${processes[@]}"; do
		if wait_for_process "$process"; then
			ERROR "$process is not running after max time"
			exit 1
		else
			INFO "$process is running"
		fi
	done
	# Wait processes to be running
	/bin/bash
}

bootstrap() {
	set -euo pipefail
	apt update -y
	DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
	apt install -y git gcc g++ curl netbase python3 libglib2.0-0 libsm6 libxrender-dev wget libxext-dev libgl1-mesa-glx socat conntrack ffmpeg apt-transport-https ca-certificates software-properties-common
	mkdir -p /etc/apt/sources.list.d

	echo "Installing GCloud"
	echo 'deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main' | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
	curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
	apt-get update -y
	apt-get install google-cloud-sdk -y

	apt clean && rm -rf /var/cache/apt/

	echo "Installing Docker"
	curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add -
	add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu focal stable"
	apt-cache policy docker-ce
	apt install -y docker-ce
	curl -L "https://github.com/docker/compose/releases/download/${DOCKER_COMPOSE_VERSION}/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
	chmod +x /usr/local/bin/docker-compose && docker-compose version

	echo "Installing Bazel"
	apt install -y apt-transport-https curl gnupg
	curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor >bazel.gpg
	mv bazel.gpg /etc/apt/trusted.gpg.d/
	echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list
	apt-get update -y
	apt install -y bazel-5.2.0
	ln -s /usr/bin/bazel-5.2.0 /usr/bin/bazel

	echo "Installing Packages"
	apt install -y tree libjpeg-dev libgeos-dev
}

"${1:-startup}" "${@:2}"
