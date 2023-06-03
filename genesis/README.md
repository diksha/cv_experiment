# Genesis

One stop for Infrastructure Automation @ Voxel

## Directory Structure
- `terraform` - Terraform Automation directory
    - `environments` - Contains the various isolated terraform setups. Please note that these environments may not always be equivalent to the standard environments like 'development', 'staging', 'production' etc.
    - `modules` - A directory for individual re-usable terraform modules. Modules starting with `base-` may use other modules from the same directory.
- `k8s` - A directory to keep track of kubernetes specific manifests and helm charts

## What is this ?

## Before you begin

You should follow the steps in the Requirements section to setup the environment to be able to use this repo.

## Types of Infrastructure Ecosystems

### `AWS: product`

This type of environments can be used to deploy the voxel perception and portal ecosystems.

It contains the following components:

- Standard VPC with 6 public and 3 private subnets. Port 80 and 443 pre-opened for public access in the public subnet.
- Standard EKS with GPU nodes and Observability
- A `runners-production` namespace with a `runners-secret` namespace that contains secrets for the perception runners.

### `AWS: platform`

This is meant to be a singleton environment for running org-wide services mostly owned by the Platform team. The workloads in this environment are considered production-critical and also should ideally be completely firewalled and protected behind a VPN. This is the base environment that needs to be setup before all others.

It contains the following components:

- Standard VPC with 6 public and 3 private subnets. Port 80 and 443 pre-opened for public access in the public subnet.
- Standard EKS with GPU nodes and Observability
- ArgoCD@EKS
- Grafana@EKS

### `AWS: ci-cd`

This module is for managing CI/CD workloads 

## Standard Modules

### `AWS: EKS`

### `AWS: Observability`

The observability module provides an ecosystem of Prometheus, Tempo, Loki and Cloudwatch to linked to platform Grafana.

It also deploys a couple of OpenTelemtry Collectors:

- **Default** - linked with our open-source stack
- **NewRelic** - Linked to our NewRelic account

## Specific Modules

### `AWS Platform - ArgoCD`

- Summary:
- [**Location**](/terraform/modules/aws/platform/modules/argo-cd)

- [**Documentation**](/terraform/modules/aws/platform/modules/argo-cd/README.md)

### `AWS Platform - Grafana`

- Summary:
- [**Location**](/terraform/modules/aws/platform/modules/eks-grafana)

- [**Documentation**](/terraform/modules/aws/platform/modules/eks-grafana/README.md)

## Requirements
For running any of these repositories locally you need some tools installed and configured
### 1. aws cli

you can run the below commands to install the cli locally

```shell
sudo apt-get install awscli
```
you can then run "aws configure" command to create a config file for your cli.

### 2. terraform
```shell
wget -O- https://apt.releases.hashicorp.com/gpg | gpg --dearmor | sudo tee /usr/share/keyrings/hashicorp-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/hashicorp.list
sudo apt update && sudo apt install terraform
```

### 3. terragrunt 
you should not use the snap version of this tool as this is outdated.
first let's install jq
```shell
sudo apt-get -y install jq
```
```shell
LATEST_URL=$(curl -sL  https://api.github.com/repos/gruntwork-io/terragrunt/releases  | jq -r '.[0].assets[].browser_download_url' | egrep 'linux.*amd64' | tail -1)
curl -sL ${LATEST_URL} > ${HOME}/bin/terragrunt
chmod +x ${HOME}/bin/terragrunt
```

### 4. golang
```shell
apt-get install golang
echo 'GOPATH=~/go' >> ~/.bashrc
source ~/.bashrc
mkdir $GOPATH
```
### 5. sops
```shell
curl -O -L -C - https://github.com/mozilla/sops/releases/download/v3.7.3/sops-v3.7.3.linux.amd64
sudo chmod +x /usr/bin/sops
sops -v
```
to use sops you also need to povide access to the key stored in the gcp kms
```shell
gcloud auth application-default login
gcloud kms keys list --location global --keyring terraform-sops
export KMS_PATH="projects/voxel-devops-production/locations/global/keyRings/terraform-sops/cryptoKeys/sops-sl1"
```
### 6. kubectl
```shell
sudo apt-get install -y ca-certificates curl
sudo curl -fsSLo /usr/share/keyrings/kubernetes-archive-keyring.gpg https://packages.cloud.google.com/apt/doc/apt-key.gpg
echo "deb [signed-by=/usr/share/keyrings/kubernetes-archive-keyring.gpg] https://apt.kubernetes.io/ kubernetes-xenial main" | sudo tee /etc/apt/sources.list.d/kubernetes.list
sudo apt-get update
sudo apt-get install -y kubectl
```
To connect to any of our EKS clusters please run the below commands while having access to the account your EKS cluster is in
```shell
aws eks --region $region update-kubeconfig --name $cluster_name
```
you can also check your current contexts and switch between diffrent clusters with the below commands
```shell
kubectl config get-contexts
kubectl config use-context CONTEXT_NAME
```
### 7. helm
```shell
curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3
chmod 700 get_helm.sh
./get_helm.sh
```

### 7. Docker
for any pulling and pushing the images we need docker installed 
```shell
sudo apt-get install ca-certificates curl gnupg lsb-release
 curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-compose-plugin
```

## Deletion of any EKS clusters
The clusters have issues with being destroyed through terragrunt so in these scenarios, you can delete the cluster through eksctl and then delete the state file through terragrunt state list|grep module."name" and then terragrunt state rm. afterward you can do terragrunt destroy -target="" for the rest of it

## Setup of providers for proper module setup
https://support.hashicorp.com/hc/en-us/articles/1500000332721-Error-Provider-configuration-not-present
