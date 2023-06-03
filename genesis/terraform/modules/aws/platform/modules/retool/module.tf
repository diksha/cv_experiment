locals {
  name      = "retool"
  namespace = local.name
}

resource "random_password" "db_password" {
  length           = 20
  special          = true
  override_special = "_-@"
}


resource "random_password" "encryption_key" {
  length           = 40
  special          = true
  override_special = "_-@"
}

resource "random_password" "jwt_secret" {
  length           = 40
  special          = true
  override_special = "_-@"
}

resource "helm_release" "main" {
  name             = local.name
  namespace        = local.namespace
  chart            = "retool"
  repository       = "https://charts.retool.com"
  version          = "4.11.6"
  wait             = true
  timeout          = 300
  create_namespace = true
  values = [templatefile("${path.module}/files/service-helm-values.yaml", {
    GOOGLE_CLIENT_ID = var.google_client_id
    })
  ]
  set_sensitive {
    name  = "config.auth.google.clientSecret"
    value = var.google_client_secret
  }
  set_sensitive {
    name  = "config.encryptionKey"
    value = random_password.encryption_key.result
  }
  set_sensitive {
    name  = "config.jwtSecret"
    value = random_password.jwt_secret.result
  }
  set_sensitive {
    name  = "config.licenseKey"
    value = var.retool_license_key
  }
  set_sensitive {
    name  = "postgresql.postgresqlPostgresPassword"
    value = random_password.db_password.result
  }
  set_sensitive {
    name  = "postgresql.postgresqlPassword"
    value = random_password.db_password.result
  }
}


resource "kubernetes_ingress_v1" "ingress" {
  metadata {
    name      = local.name
    namespace = local.namespace
  }

  spec {
    ingress_class_name = "nginx"
    rule {
      host = "retool.voxelplatform.com"
      http {
        path {
          backend {
            service {
              name = local.name
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

module "postgres_shared" {
  source           = "../../../shared/eks-postgres"
  db_identifier    = "${local.name}-shared"
  namespace        = local.namespace
  cluster_name     = var.eks_cluster_name
  account_id       = var.account_id
  create_namespace = false
  replica_count    = 1
}