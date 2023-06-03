#!/usr/bin/env bash
set -euo pipefail

# distribution=$(. /etc/os-release;echo $ID$VERSION_ID) && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.repo | sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo
# sudo yum update -y
# sudo yum-config-manager --enable libnvidia-container-experimental
# sudo yum install -y https://s3.amazonaws.com/ec2-downloads-windows/SSMAgent/latest/linux_amd64/amazon-ssm-agent.rpm
# sudo yum install -y nvidia-docker2 nvidia-container-toolkit
# # ec2-35-88-216-245.us-west-2.compute.amazonaws.com

# wget https://us.download.nvidia.com/tesla/515.48.07/nvidia-driver-local-repo-rhel9-515.48.07-1.0-1.x86_64.rpm
# sudo yum install -y nvidia-driver-local-repo-rhel9-515.48.07-1.0-1.x86_64.rpm
# sudo systemctl enable docker || echo "FAILED: sudo systemctl enable docker"
# sudo systemctl enable sshd || echo "FAILED: sudo systemctl enable sshd"

sudo yum install -y gcc kernel-devel-$(uname -r)
BASE_URL=https://us.download.nvidia.com/tesla
DRIVER_VERSION=510.47.03
curl -fSsl -O $BASE_URL/$DRIVER_VERSION/NVIDIA-Linux-x86_64-$DRIVER_VERSION.run
sudo CC=/usr/bin/gcc10-cc sh NVIDIA-Linux-x86_64-$DRIVER_VERSION.run -s
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.repo | sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo
sudo yum install nvidia-container-toolkit nvidia-container-runtime nvidia-docker2 -y

sudo systemctl enable docker
