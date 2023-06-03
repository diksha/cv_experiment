
# resource "helm_release" "otel_collector" {
#   name             = "otel-collector"
#   namespace        = "observability"
#   create_namespace = true
#   repository       = "https://open-telemetry.github.io/opentelemetry-helm-charts"
#   chart            = "opentelemetry-collector"
#   version          = "0.36.3"
#   values = [
#     templatefile("${path.module}/ymls/otel-collector-default.yml", {
#       CLUSTER_NAME          = var.cluster_name
#       AWS_REGION            = var.aws_region
#       OTLP_TRACING_ENDPOINT = "tempo-tempo-distributed-distributor:55680"
#     })
#   ]
# }


# resource "helm_release" "promtail" {
#   name             = "promtail"
#   namespace        = helm_release.observability_stack.namespace
#   create_namespace = true
#   repository       = "https://grafana.github.io/helm-charts"
#   chart            = "promtail"
#   version          = "6.5.1"
#   values = [
#   templatefile("${path.module}/ymls/promtail.yml", {
#       LOKI_URL = "http://loki-distributed-gateway/loki/api/v1/push"
#       LOKI_TENANT_ID = "${var.account_id}-${var.cluster_name}"
#     })
#   ]
# }

# resource "helm_release" "otel_operator" {
#   name             = "otel-operator"
#   namespace        = "observability"
#   create_namespace = true
#   repository       = "https://open-telemetry.github.io/opentelemetry-helm-charts"
#   chart            = "opentelemetry-operator"
#   version          = "0.15.0"
# }

resource "helm_release" "otel_operator" {
  name             = "otel-operator"
  namespace        = "observability"
  create_namespace = true
  chart            = "https://github.com/itscontained/charts/releases/download/raw-v0.2.5/raw-v0.2.5.tgz"
  values = [
    templatefile("${path.module}/ymls/otel-operator.yml", {
    })
  ]
}

resource "helm_release" "otel_collector_via_crd" {
  name             = "otel-collector"
  namespace        = "observability"
  create_namespace = true
  chart            = "https://github.com/itscontained/charts/releases/download/raw-v0.2.5/raw-v0.2.5.tgz"
  values = [
    templatefile("${path.module}/ymls/otel-collector-default-via-crd.yml", {
      CLUSTER_NAME            = var.cluster_name
      AWS_REGION              = var.aws_region
      OTLP_TRACING_ENDPOINT   = "tempo-tempo-distributed-distributor:55680"
      LOKI_BASE_ENDPOINT      = "http://loki-distributed-distributor:3100"
      LOKI_TENANT_ID          = "${var.account_id}-${var.cluster_name}"
      COLLECTOR_NAME          = "default"
      NAMESPACE               = "observability"
      SERVICE_ACCOUNT_NAME    = kubernetes_service_account.observability_aws.metadata[0].name
      PROMETHEUS_RELEASE_NAME = helm_release.observability_stack.name
    })
  ]

  depends_on = [helm_release.otel_operator]
}

resource "kubernetes_labels" "prometheus_labels_for_otel_default" {
  api_version = "v1"
  kind        = "Service"
  metadata {
    name      = "default-collector"
    namespace = helm_release.otel_collector_via_crd.namespace
  }
  labels = {
    prometheus = "default"
  }
}