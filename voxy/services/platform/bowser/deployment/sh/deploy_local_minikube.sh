#!/bin/bash
MINIKUBE_MEMORY=8192
MINIKUBE_CPU=4
MINIKUBE_PROFILE=bowser
KUBERNETERS_VERSION="v1.25.3"
DOCKER_INTRA_DNS="docker.local"
DEV_BOX_LOCAL_IP="192.168.58.1"

echo "----------------"
echo "check installation kubectl"
if [ -x "$(command -v kubectl)" ]; then
	echo "kubectl is installed"
else
	curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
	sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
	kubectl version --client
fi

echo "----------------"
echo "check installation helm"
if [ -x "$(command -v helm)" ]; then
	echo "helm is installed"
else
	curl https://baltocdn.com/helm/signing.asc | gpg --dearmor | sudo tee /usr/share/keyrings/helm.gpg >/dev/null
	sudo apt-get install apt-transport-https --yes
	echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/helm.gpg] https://baltocdn.com/helm/stable/debian/ all main" | sudo tee /etc/apt/sources.list.d/helm-stable-debian.list
	sudo apt-get update
	sudo apt-get install helm
fi

echo "----------------"
echo "check installation minikube"
if [ -x "$(command -v minikube)" ]; then
	echo "minikube is installed"
	mkdir -p ~/.minikube/files/etc
	echo $DEV_BOX_LOCAL_IP $DOCKER_INTRA_DNS >~/.minikube/files/etc/hosts
else
	curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
	sudo install minikube-linux-amd64 /usr/local/bin/minikube
	mkdir -p ~/.minikube/files/etc
	echo "$IP" $DOCKER_INTRA_DNS >~/.minikube/files/etc/hosts
fi

echo "----------------"
echo "start minikube with cpu:$MINIKUBE_CPU memory:$MINIKUBE_MEMORY"
minikube -p $MINIKUBE_PROFILE start --mount-string="/home/antoine/.aws:/root/.aws" --mount --memory $MINIKUBE_MEMORY --cpus $MINIKUBE_CPU --kubernetes-version=$KUBERNETERS_VERSION --insecure-registry="$DOCKER_INTRA_DNS:5000"
minikube profile $MINIKUBE_PROFILE
minikube addons enable ingress
echo "----------------"
echo "Here is your minikube profile list bowser operator will be installed on profile $MINIKUBE_PROFILE"
minikube profile list
echo "----------------"
echo "Deploy Bowser Kube Operator"
kubectl create -f https://github.com/jetstack/cert-manager/releases/download/v1.8.2/cert-manager.yaml
helm repo add flink-operator-repo https://downloads.apache.org/flink/flink-kubernetes-operator- <OPERATOR-VERSION >/
echo "Installing FKO with these configuration"
cat ../kube/bowser_kubernetes_operator_values_config.yaml
helm install -f services/platform/bowser/deployment/sh/bowser_kubernetes_operator_values_config.yaml --set metrics.port=9999 flink-kubernetes-operator flink-operator-repo/flink-kubernetes-operator

echo "----------------"
echo "Create Kube serviceaccount and clusterrolebinding"
kubectl create serviceaccount bowser-testing
kubectl create clusterrolebinding bowser-role-binding-bowser --clusterrole=edit --serviceaccount=default:bowser-testing

echo "----------------"
echo "Create minikube prometheus"
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prometheus prometheus-community/kube-prometheus-stack
echo "Grafana Login : " && kubectl get secret prometheus-grafana -o jsonpath="{.data.admin-user}" | base64 --decode echo
echo "Grafana Password : " && kubectl get secret prometheus-grafana -o jsonpath="{.data.admin-password}" | base64 --decode echo
echo "Started to Forward port 3000 ( grafana web ui ). Ssh tunel in order to get acces to http://localhost:3000"
kubectl create -f services/platform/bowser/deployment/kube/minikube_prometheus_podmonitor_for_bowser.yaml
kubectl port-forward deployment/prometheus-grafana 3000 &

echo "----------------"
echo "Create minikube dashboard"
minikube -p MINIKUBE_PROFILE dashboard

echo "Create minikube OpenTelemetry Operator and collector"
kubectl apply -f https://github.com/open-telemetry/opentelemetry-operator/releases/latest/download/opentelemetry-operator.yaml
kubectl apply -f services/platform/bowser/deployment/kube/minikube_opentelemetry_collector.yaml

echo "----------------"
echo "generate minikube dashboard"
minikube -p MINIKUBE_PROFILE dashboard --url
