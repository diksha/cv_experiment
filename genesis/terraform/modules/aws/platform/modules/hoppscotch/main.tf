resource "helm_release" "primary" {
  name             = var.deplomyment_identifier
  namespace        = var.deplomyment_identifier
  chart            = "${path.module}/../../../shared/charts/generic"
  wait             = true
  timeout          = 300
  replace          = false
  create_namespace = true
  values = [
    templatefile("${path.module}/files/service-helm-values.yaml",
      {
        DOMAIN                = var.domain
    })
  ]
}

resource "kubernetes_ingress_v1" "opencov_ingress" {
  metadata {
    name = var.deplomyment_identifier
    namespace = var.deplomyment_identifier
  }

  spec {
    ingress_class_name = "nginx"
    rule {
      host = "apidev.voxelplatform.com"
      http {
        path {
          backend {
            service {
              name = "${var.deplomyment_identifier}-generic"
            port {
              number = 80
            }
          }
          }
          path = "/"
        }
      }
    }
    rule {
      host = "apidev.private.voxelplatform.com"
      http {
        path {
          backend {
            service {
              name = "${var.deplomyment_identifier}-generic"
            port {
              number = 80
            }
          }
          }
          path = "/"
        }
      }
    }
  }
}