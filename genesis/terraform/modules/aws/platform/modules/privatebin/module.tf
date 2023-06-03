resource "helm_release" "main" {
  name             = "privatebin"
  namespace        = "privatebin"
  chart            = "privatebin"
  repository       = "https://privatebin.github.io/helm-chart"
  version          = "0.17.0"
  wait             = true
  timeout          = 600
  create_namespace = true
  values = [templatefile("${path.module}/files/service-helm-values.yaml", {
    })
  ]
}
