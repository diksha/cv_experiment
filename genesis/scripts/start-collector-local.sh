#!/bin/bash
set -euo pipefail


docker rm --force otel_collector_local ||:
docker run -p 4317:4317 \
    -it --name otel_collector_local --restart=always \
    -v $PWD/testdata/logs:/data \
    -v $PWD/otel-collector-config.yaml:/etc/otel-collector-config.yaml \
    otel/opentelemetry-collector-contrib:0.56.0 \
    --config=/etc/otel-collector-config.yaml

echo """

Please set the following env vars in your app to get started locally:

export OTEL_TRACES_EXPORTER=otlp
export OTEL_METRICS_EXPORTER=otlp
export OTEL_LOGS_EXPORTER=otlp
export OTEL_SERVICE_NAME=svc-name
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317


Run this command to see the collector logs:

docker logs -f otel_collector_local
"""
