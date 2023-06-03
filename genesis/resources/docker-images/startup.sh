#!/bin/bash
set -euo pipefail

start() {
	/bin/bash
}

bootstrap() {
	apt update -y
	DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
	apt install -y zsh git gcc g++ curl netbase python3 python3-pip libglib2.0-0 libsm6 libxrender-dev wget libxext-dev libgl1-mesa-glx socat conntrack ffmpeg apt-transport-https ca-certificates software-properties-common libgeos-dev
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
}

"${1:-start}" "${@:2}"
