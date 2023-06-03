resource "helm_release" "primary" {
  name             = var.deplomyment_identifier
  namespace        = var.deplomyment_identifier
  repository       = "https://helm.buildbuddy.io"
  chart            = "buildbuddy"
  version          = "0.0.126"
  wait             = false
  timeout          = 300
  replace          = false
  create_namespace = true
  values           = [file("${path.module}/files/service-helm-values.yaml")]

  set_sensitive {
    name  = "mysql.mysqlPassword"
    value = random_password.mysql_password.result
  }
}

resource "helm_release" "mysql" {
  name             = "${var.deplomyment_identifier}-mysql"
  namespace        = var.deplomyment_identifier
  repository       = "https://charts.bitnami.com/bitnami"
  chart            = "mysql"
  version          = "9.3.4"
  wait             = false
  timeout          = 300
  replace          = false
  create_namespace = true
  values = [yamlencode({
    auth = {
      rootPassword = random_password.mysql_password.result
      database     = "buildbuddy"
      username     = "voxel"
      password     = random_password.mysql_password.result
    }
  })]
}

resource "random_password" "mysql_password" {
  special = false
  length  = 16
}


resource "kubernetes_ingress_v1" "builbuddy_ingress" {
  metadata {
    name      = "buildbuddy"
    namespace = "buildbuddy"
  }

  spec {
    ingress_class_name = "nginx"
    rule {
      host = "buildbuddy.voxelplatform.com"
      http {
        path {
          backend {
            service {
              name = "buildbuddy"
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
      host = "buildbuddy.private.voxelplatform.com"
      http {
        path {
          backend {
            service {
              name = "buildbuddy"
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
