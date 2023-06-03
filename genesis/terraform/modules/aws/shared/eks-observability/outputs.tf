data "kubernetes_service" "loki" {
  metadata {
    name      = "${helm_release.loki_distributed.name}-gateway"
    namespace = helm_release.loki_distributed.namespace
  }
  depends_on = [helm_release.loki_distributed]
}


data "kubernetes_service" "prometheus" {
  metadata {
    name      = "${helm_release.observability_stack.name}-kube-p-prometheus"
    namespace = helm_release.observability_stack.namespace
  }
  depends_on = [helm_release.observability_stack]
}


data "kubernetes_service" "tempo" {
  metadata {
    name      = "${helm_release.tempo.name}-tempo-distributed-gateway"
    namespace = helm_release.tempo.namespace
  }
  depends_on = [helm_release.tempo]
}

data "kubernetes_service" "alertmanager" {
  metadata {
    name      = "${helm_release.observability_stack.name}-kube-p-alertmanager"
    namespace = helm_release.observability_stack.namespace
  }
  depends_on = [
    helm_release.observability_stack
  ]
}
