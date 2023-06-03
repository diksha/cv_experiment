
resource "helm_release" "main" {
  name             = "flink-operator"
  namespace        = "flink-operator"
  chart            = "flink-kubernetes-operator"
  repository       = "https://downloads.apache.org/flink/flink-kubernetes-operator-1.3.1/"
  version          = "1.3.1"
  wait             = true
  timeout          = 300
  create_namespace = true
  values = [templatefile("${path.module}/files/service-helm-values.yaml", {
    NAMESPACE_TO_WATCH = var.namespace_to_watch
    })
  ]
}