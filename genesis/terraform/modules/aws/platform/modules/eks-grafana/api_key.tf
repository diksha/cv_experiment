provider "grafana" {
  url    = "https://${var.domain}/"
  auth   = "admin:${random_password.admin_password.result}"
}

resource "grafana_api_key" "tf" {
  name = "terraform"
  role = "Admin"
  depends_on = [
    helm_release.grafana
  ]
}

data "kubernetes_service" "grafana" {
  metadata {
    name      = "grafana"
    namespace = helm_release.grafana.namespace
  }
}