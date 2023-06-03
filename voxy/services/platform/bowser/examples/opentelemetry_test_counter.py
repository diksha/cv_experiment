import os
import time

from lib.infra.metrics.constants import Environment
from lib.infra.metrics.register_metrics import RegisterMetrics

# TO TEST :
# Requirement : sh services/platform/bowser/deployment/sh/deploy_local_minikube.sh
# 1 = kubectl port-forward svc/bowser-minikube-opentelemetry-collector 4317 => OTCOL GRPC TO PUSH
# 2 = kubectl port-forward deployment/prometheus-grafana 3000 => Graphana to look at
# 3 = kubectl port-forward svc/bowser-minikube-opentelemetry-collector  8889
#   => Debug to look at your metrics (http://localhost:8889/metrics)
# 4 = bazel run  //services/platform/bowser/engine/tests:test
# 5 = From your laptop browser
#        - sh services/platform/bowser/deployment/sh/minikube_tunel_ssh.sh.sh
#        - connect to : http://localhost:8889/metrics to see your metric in real time
#        - connect to : http://localhost:3000/explore to see your metric on grafana


os.environ["OTEL_EXPORTER_OTLP_METRICS_ENDPOINT"] = "http://localhost:4317"
os.environ["OTEL_METRIC_EXPORT_INTERVAL"] = "1"

if __name__ == "__main__":
    print("START PUSHING ?")
    metrics = RegisterMetrics(
        service_name="bowser_counter_test",
        metric_names=["counter_test"],
        environment=Environment.DEVELOPMENT,
        camera_uuid=None,
    )

    while True:
        metrics.increment_metric_counter_with_attributes(
            "counter_test", 1, {"label_key": "label_value"}
        )
        print("----")
        time.sleep(1)
