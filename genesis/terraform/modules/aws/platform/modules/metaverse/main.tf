locals {
  db_name = var.environment == "production" ? var.db_identifier : "${var.db_identifier}-${var.environment}"
  service_name = "${local.db_name}-neo4j"
}


module "neo4j" {
  providers = {
    aws = aws
  }
  source                               = "../../../shared/neo4j"
  account_id                           = var.account_id
  cluster_name                         = var.cluster_name
  db_identifier                        = local.db_name
}

resource "kubernetes_ingress_v1" "ingress" {
  metadata {
    name = local.service_name
    namespace = local.db_name
  }
  spec {
    ingress_class_name = "nginx"
    rule {
      host = "${local.service_name}.voxelplatform.com"
      http {
        path {
          backend {
            service {
              name = local.service_name
            port {
              number = 7474
            }
          }
          }
          path = "/"
        }
      }
    }
    rule {
      host = "${local.service_name}.private.voxelplatform.com"
      http {
        path {
          backend {
            service {
              name = local.service_name
            port {
              number = 7474
            }
          }
          }
          path = "/"
        }
      }
    }
}
}
