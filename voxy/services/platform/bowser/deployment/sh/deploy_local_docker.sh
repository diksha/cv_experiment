#!/bin/bash

# trunk-ignore-all(shellcheck):
DOCKER_VERSION=5:20.10.23~3-0~ubuntu-focal
DOCKER_INTRA_DNS="docker.local"
ETC_HOSTS_PATH="/etc/hosts"
DOCKER_DEAMON_JSON_PATH="/etc/docker/daemon.json"

echo "----------------"
echo "check installation docker"
if [ -x "$(command -v docker)" ]; then
	echo "Docker is installed"
else
	echo "Install docker"
	sudo apt-get update
	sudo apt-get install ca-certificates curl gnupg >/dev/null
	sudo mkdir -m 0755 -p /etc/apt/keyrings
	curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
	echo \
		"deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
      "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" |
		sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
	sudo apt-get -y install docker-ce=$DOCKER_VERSION
	sleep 20s
	sudo apt-get -y install docker-ce-cli=$DOCKER_VERSION
	sudo apt-get -y install containerd.io
	sudo apt-get -y install docker-buildx-plugin
	sudo apt-get -y install docker-compose-plugin
	echo "Make docker usable without sudo ( without that = issue with bazel later )"
	sudo chown "$USER" /var/run/docker.sock
	sudo chown -R "$USER" /etc/docker
	echo "docker installed"
fi

echo "----------------"

echo "Check your dev hosts file "
sleep 1s
echo "Executing command : cat $ETC_HOSTS_PATH"
sleep 3s
cat $ETC_HOSTS_PATH
echo "----------------"
while true; do
	read -p "Are you seeing $DOCKER_INTRA_DNS linked to localhost or 127.0.0.1 ? (Y/N)" yn
	case $yn in
	[Yy]*)
		echo "Continue installation"
		break
		;;
	[Nn]*)
		echo "Then we add that right away"
		sudo echo "127.0.0.1 $DOCKER_INTRA_DNS" >>$ETC_HOSTS_PATH
		break
		;;
	*) echo "Please answer yes or no." ;;
	esac
done

echo "----------------"
echo "Check Docker deamon JSON"
sleep 1s
if [ -e $DOCKER_DEAMON_JSON_PATH ]; then
	echo "Executing command : cat $DOCKER_DEAMON_JSON_PATH"
	sleep 3s
	cat $DOCKER_DEAMON_JSON_PATH
	while true; do
		read -p "Are you seeing a JSON with an insecure-registries array field with inside your http://$DOCKER_INTRA_DNS:500 ? (Y/N)" yn
		case $yn in
		[Yy]*)
			echo "Continue installation"
			break
			;;
		[Nn]*)
			echo "Then we add that right away"
			sudo touch $DOCKER_DEAMON_JSON_PATH
			sudo chmod 777 /etc/docker/daemon.json
			sleep 1s
			sudo echo "{\"insecure-registries\" : [\"http://$DOCKER_INTRA_DNS:5000\"]}" >>$DOCKER_DEAMON_JSON_PATH
			echo "Done !"
			break
			;;
		*) echo "Please answer yes or no." ;;
		esac
	done
else
	echo "Looks like your docker deamon json is not found"
	echo "I am creating that for you buddy"
	sleep 1s
	sudo touch $DOCKER_DEAMON_JSON_PATH
	sudo chmod 777 /etc/docker/daemon.json
	sleep 1s
	sudo echo "{\"insecure-registries\" : [\"http://$DOCKER_INTRA_DNS:5000\"]}" >>$DOCKER_DEAMON_JSON_PATH
	echo "Done !"
fi

echo "----------------"
echo "Run Docker registry on port 5000 with REGISTRY_STORAGE_DELETE_ENABLED turn on true"
docker run -e REGISTRY_STORAGE_DELETE_ENABLED=true -d -p 5000:5000 --restart=always --name registry registry:2
sleep 1s

echo "----------------"
echo "restart Docker (30second)"
sudo systemctl daemon-reload
sudo systemctl stop docker
sudo systemctl start docker

echo "----------------"
echo "Check docker registry"
sleep 1s
echo "Executing command : docker info"
sleep 3s
docker info
while true; do
	read -p "Are you seeing your your insecure registry on the docker info command ? (Y/N)" yn
	case $yn in
	[Yy]*)
		echo "Good Good!"
		break
		;;
	[Nn]*)
		echo "You have a problem houston"
		exit
		;;
	*) echo "Please answer yes or no." ;;
	esac
done

echo "----------------"
echo "Push the flink docker image to your local registry"
/home/antoine/workspace/Voxel-Monolithic/bazel run //services/platform/bowser/processors:push_minikube_local

echo "----------------"
echo "At this time you should be able to get access to your local docker registry and see your image pushed by bazel "
sleep 1s
echo "Executing command : curl -X  GET http://$DOCKER_INTRA_DNS:5000/v2/services/platform/bowser/tags/list"
sleep 3s
curl -X GET http://docker.local:5000/v2/services/platform/bowser/tags/list #curl respo
while true; do
	read -p "Are you seeing your flink docker image ? (Y/N)" yn
	case $yn in
	[Yy]*)
		echo "OMG that's a miracle ! Then we continue my man"
		break
		;;
	[Nn]*)
		echo "You have a problem houston"
		exit
		;;
	*) echo "Please answer yes or no." ;;
	esac
done

echo "----------------"
echo "At this time you should be to docker pull you image from your local registry"
sleep 1s
echo "Executing command : docker pull  $DOCKER_INTRA_DNS:5000/services/platform/bowser:latest"
sleep 3s
docker pull $DOCKER_INTRA_DNS:5000/services/platform/bowser:latest
while true; do
	read -p "Are you seeing docker pulling your docker image ? (Y/N)" yn
	case $yn in
	[Yy]*)
		echo "that's a good news"
		break
		;;
	[Nn]*)
		echo "You have a problem houston"
		exit
		;;
	*) echo "Please answer yes or no." ;;
	esac
done
