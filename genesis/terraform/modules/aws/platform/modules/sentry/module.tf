resource "random_password" "admin_password" {
  length           = 20
  special          = true
  override_special = "_-@"
}

resource "random_password" "db_password" {
  length           = 20
  special          = true
  override_special = "_-@"
}

# After deploying enable autovacuum on the nodestore table to allow db cleanups.
# ALTER TABLE nodestore_node SET (autovacuum_enabled = on);
resource "helm_release" "main" {
  name             = "sentry"
  namespace        = "sentry"
  chart            = "sentry"
  repository       = "https://sentry-kubernetes.github.io/charts"
  version          = "17.7.0"
  wait             = true
  timeout          = 300
  create_namespace = true
  values = [templatefile("${path.module}/files/service-helm-values.yaml", {
    GOOGLE_CLIENT_ID = var.google_client_id
    })
  ]
  set_sensitive {
    name = "google.clientSecret"
    value = var.google_client_secret
  }
  set_sensitive {
    name = "user.password"
    value = random_password.admin_password.result
  }
  set_sensitive {
    name = "postgresql.postgresqlPostgresPassword"
    value = random_password.db_password.result
  }
  set_sensitive {
    name = "postgresql.postgresqlPassword"
    value = random_password.db_password.result
  }
  set_sensitive {
    name = "slack.clientId"
    value = var.slack.client_id
  }
  set_sensitive {
    name = "slack.clientSecret"
    value = var.slack.client_secret
  }
  set_sensitive {
    name = "slack.signingSecret"
    value = var.slack.signing_secret
  }
}

resource "kubernetes_ingress_v1" "sentry_ingress" {
  metadata {
    name = "sentry"
    namespace = "sentry"
  }

  spec {
    ingress_class_name = "nginx"
    rule {
      host = "sentry.voxelplatform.com"
      http {
        path {
          backend {
            service {
              name = "sentry-nginx"
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
      host = "sentry.private.voxelplatform.com"
      http {
        path {
          backend {
            service {
              name = "sentry-nginx"
            port {
              number = 80
            }
          }
          }
          path = "/"
        }
      }
    }
    dynamic rule {
      # Portal Frontend Projects.
      for_each = ["/api/4/envelope/", "/api/4/store/", "/api/5/envelope/", "/api/5/store/"]
      content {
        host = "st.public.voxelplatform.com"
        http {
          path {
            backend {
              service {
                name = "sentry-nginx"
              port {
                number = 80
              }
            }
            }
            path = rule.value
            path_type = "Exact"
          }
        }
      }
    }
  }
}
