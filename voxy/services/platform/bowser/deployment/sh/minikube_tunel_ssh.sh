#!/bin/sh
MINIKUBE_DASHBOARD=$1
BOWSER_PROCESSOR_NAME=$2
echo "Minikube DashBoard => http://127.0.0.1:$MINIKUBE_DASHBOARD/api/v1/namespaces/kubernetes-dashboard/services/http:kubernetes-dashboard:/proxy/"
echo "grafana DashBoard => http://127.0.0.1:3000"
echo "Bowser WebUi DashBoard => http://localhost:8080/default/$BOWSER_PROCESSOR_NAME" | pbcopy
echo "Bowser Dashboard is also on your clipboard"
#8080 => Flink WebUI ingress
#3000 => Grafana. Need  kubectl port-forward deployment/prometheus-grafana 3000
#9249 => Flink JobManager/Taskmanager system metric. Need kubectl port-forward poc-bowser-count-incident-taskmanager-1-1  9249
#8889 => OTCOL. Need kubectl port-forward svc/bowser-minikube-opentelemetry-collector  8889
ssh -N -L "$MINIKUBE_DASHBOARD":127.0.0.1:"$MINIKUBE_DASHBOARD" -L 8080:192.168.49.2:80 -L 3000:localhost:3000 -L 9249:localhost:9249 -L 8889:localhost:8889 "$USER"@"$USER".devbox.voxelplatform.com
