resource "helm_release" "primary" {
  name             = var.deplomyment_identifier
  namespace        = var.target_namespace
  chart            = "${path.module}/../charts/generic"
  wait             = true
  timeout          = 120
  replace          = false
  create_namespace = true
  values = [templatefile("${path.module}/files/service-helm-values.yaml", {
    DEPLOYMENT_IDENTIFIER = var.deplomyment_identifier
    })
  ]
  set_sensitive {
    name  = "env.secret.TUNNEL_TOKEN"
    value = var.tunnel_token
  }
}
