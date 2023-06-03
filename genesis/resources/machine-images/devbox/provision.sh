#!/usr/bin/env bash
set -euo pipefail

apt update -y
DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
apt install -y openssh-server sudo vim iproute2 zsh git gcc g++ curl netbase python3 libglib2.0-0 libsm6 libxrender-dev wget libxext-dev libgl1-mesa-glx socat conntrack ffmpeg apt-transport-https ca-certificates software-properties-common libgeos-dev
mkdir -p /etc/apt/sources.list.d

echo 'deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main' | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
apt-get update -y
apt-get install google-cloud-sdk -y


# Minikube
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube_latest_amd64.deb
dpkg -i minikube_latest_amd64.deb && rm minikube_latest_amd64.deb

# Docker Compose
curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose && docker-compose version

snap install kubectl --classic

# CLEANUP
apt clean && rm -rf /var/cache/apt/

systemctl enable docker

mv /tmp/devbox.sh /devbox.sh && chmod +rx /devbox.sh