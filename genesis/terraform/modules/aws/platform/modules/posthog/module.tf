resource "aws_ecr_repository" "posthog" {
  name = "posthog"
  image_tag_mutability = "IMMUTABLE"

}

resource "aws_ecr_lifecycle_policy" "ecr_policy" {
  repository = aws_ecr_repository.posthog.name
  policy = <<EOF
{
    "rules": [
        {
            "rulePriority": 1,
            "description": "Keep last 5 images",
            "selection": {
                "tagStatus": "any",
                "countType": "imageCountMoreThan",
                "countNumber": 5
            },
            "action": {
                "type": "expire"
            }
        }
    ]
}
EOF
}

resource "helm_release" "main" {
  name             = "posthog"
  namespace        = "posthog"
  chart            = "posthog"
  repository       = "https://posthog.github.io/charts-clickhouse/"
  version          = "30.2.7"
  wait             = true
  timeout          = 600
  create_namespace = true
  values = [templatefile("${path.module}/files/service-helm-values.yaml", {
    GOOGLE_CLIENT_ID = var.google_client_id
    SITE_URL  = "https://posthog.voxelplatform.com"
    })
  ]
  set_sensitive{
    name = "web.env[1].value"
    value = var.google_client_secret
  }
  set_sensitive{
    name = "env[1].value"
    value = var.google_client_secret
  }
}

# Posthog Chart has issue in the events-service yaml file where they have an extra space
# at the beginning in the annotations key value line, causing bad indent and yaml to fail.
# That's why we put that as ClusterIP and create our own LoadBalancer.

# resource "kubernetes_service" "posthog_load_balancer" {
#   metadata {
#     name = "posthog-load-balancer"
#     namespace = "posthog"
#     annotations = {
#       "service.beta.kubernetes.io/aws-load-balancer-scheme" = "internal"
#       "service.beta.kubernetes.io/aws-load-balancer-type" = "external"
#       "service.beta.kubernetes.io/aws-load-balancer-nlb-target-type" = "ip"
#       "service.beta.kubernetes.io/load-balancer-source-ranges" = "10.0.0.0/8"
#       "service.beta.kubernetes.io/aws-load-balancer-additional-resource-tags" = "map-migrated=d-server-00swbp99drezfh"
#     }
#   }
#   spec {
#     selector = {
#       "app" = "posthog"
#       "role" = "events"
#     }
#     port {
#       port        = 80
#       target_port = 8000
#     }
#     type = "LoadBalancer"
#   }
# }




resource "kubernetes_ingress_v1" "posthog_ingress" {
  metadata {
    name = "posthog"
    namespace = "posthog"
  }
  spec {
    ingress_class_name = "nginx"
    rule {
      host = "posthog.private.voxelplatform.com"
      http {
        path {
          backend {
            service {
              name = "posthog-events"
            port {
              number = 8000
            }
          }
          }
          path = "/"
        }
      }
    }
    rule {
      host = "posthog.voxelplatform.com"
      http {
        path {
          backend {
            service {
              name = "posthog-events"
            port {
              number = 8000
            }
          }
          }
          path = "/"
        }
      }
    }
    dynamic rule {
      for_each = ["/batch", "/decide", "/capture", "/e", "/engage", "/s", "/static/array.js", "/track"]
      content {
        host = "ph.public.voxelplatform.com"
        http {
          path {
            backend {
              service {
                name = "posthog-events"
              port {
                number = 8000
              }
            }
            }
            path = rule.value
            path_type = "Prefix"
          }
        }
      }
    }
  }
}

resource "helm_release" "development" {
  name             = "posthog"
  namespace        = "posthog-development"
  chart            = "posthog"
  repository       = "https://posthog.github.io/charts-clickhouse/"
  version          = "30.2.7"
  wait             = true
  timeout          = 600
  create_namespace = true
  values = [templatefile("${path.module}/files/service-helm-values.yaml", {
    GOOGLE_CLIENT_ID = var.google_client_id
    SITE_URL  = "https://posthog-development.voxelplatform.com"
    })
  ]
  set_sensitive{
    name = "web.env[1].value"
    value = var.google_client_secret
  }
  set_sensitive{
    name = "env[1].value"
    value = var.google_client_secret
  }
}

resource "kubernetes_ingress_v1" "posthog_development_ingress" {
  metadata {
    name = "posthog"
    namespace = "posthog-development"
  }
  spec {
    ingress_class_name = "nginx"
    rule {
      host = "posthog-development.private.voxelplatform.com"
      http {
        path {
          backend {
            service {
              name = "posthog-events"
            port {
              number = 8000
            }
          }
          }
          path = "/"
        }
      }
    }
    rule {
      host = "posthog-development.voxelplatform.com"
      http {
        path {
          backend {
            service {
              name = "posthog-events"
            port {
              number = 8000
            }
          }
          }
          path = "/"
        }
      }
    }
    dynamic rule {
      for_each = ["/batch", "/decide", "/capture", "/e", "/engage", "/s", "/static/array.js", "/track"]
      content {
        host = "ph-development.public.voxelplatform.com"
        http {
          path {
            backend {
              service {
                name = "posthog-events"
              port {
                number = 8000
              }
            }
            }
            path = rule.value
            path_type = "Prefix"
          }
        }
      }
    }
  }
}