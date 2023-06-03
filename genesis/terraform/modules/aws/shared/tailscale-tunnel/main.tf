locals {
  oidc_provider = replace(data.aws_eks_cluster.cluster.identity[0].oidc[0].issuer, "https://", "")
}


resource "helm_release" "primary" {
  name             = var.deplomyment_identifier
  namespace        = var.target_namespace
  chart            = "${path.module}/../charts/generic"
  version          = "0.1.0"
  wait             = true
  timeout          = 120
  replace          = false
  create_namespace = true
  values = [templatefile("${path.module}/files/service-helm-values.yaml", {
    DEPLOYMENT_IDENTIFIER = var.deplomyment_identifier
    })
  ]
  set_sensitive {
    name  = "env.secret.TS_AUTH_KEY"
    value = var.tailscale_auth_key
  }
}
