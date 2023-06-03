locals {
  name = "foxglove"
  namespace = local.name
  hostname = "${local.name}.${var.root_domain}"
}


resource "helm_release" "main" {
  name             = local.name
  namespace        = local.namespace
  chart            = "${path.module}/../../../shared/charts/generic"
  wait             = true
  timeout          = 300
  create_namespace = true
  values = [templatefile("${path.module}/files/service-helm-values.yaml", {
    })
  ]
}

resource "kubernetes_ingress_v1" "ingress" {
  metadata {
    name = local.name
    namespace = local.namespace
  }

  spec {
    ingress_class_name = "nginx"
    rule {
      host = local.hostname
      http {
        path {
          backend {
            service {
              name = "${local.name}-generic"
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