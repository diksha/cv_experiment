resource "random_password" "http_session" {
  length           = 50
  special          = false
}

resource "random_password" "auth_token" {
  length           = 50
  special          = false
}

resource "random_string" "api_server_access_key" {
  length           = 20
  special          = false
}

resource "random_password" "api_server_secret_key" {
  length           = 50
  special          = false
}

resource "random_string" "tests_access_key" {
  length           = 20
  special          = false
}

resource "random_password" "tests_secret_key" {
  length           = 50
  special          = false
}

resource "random_string" "default_company" {
  length           = 32
  special          = false
}


resource "helm_release" "clearml" {
  name             = "clearml"
  namespace        = kubernetes_namespace.clearml.metadata[0].name
  chart            = "clearml"
  repository       = "https://allegroai.github.io/clearml-helm-charts"
  version          = "4.3.0"
  wait             = true
  timeout          = 300
  create_namespace = true
  values = [templatefile("${path.module}/files/service-helm-values.yaml", {
    })
  ]
  set_sensitive {
    name = "clearml.defaultCompany"
    value = random_string.default_company.result
  }
  set_sensitive {
    name = "secret.httpSession"
    value = random_password.http_session.result
  }
  set_sensitive {
    name = "secret.authToken"
    value = random_password.auth_token.result
  }
  set_sensitive {
    name = "secret.credentials.apiserver.accessKey"
    value = random_string.api_server_access_key.result
  }
  set_sensitive {
    name = "secret.credentials.apiserver.secretKey"
    value = random_password.api_server_secret_key.result
  }
  set_sensitive {
    name = "secret.credentials.tests.accessKey"
    value = random_string.tests_access_key.result
  }
  set_sensitive {
    name = "secret.credentials.tests.secretKey"
    value = random_password.tests_secret_key.result
  }
}

resource "kubernetes_namespace" "clearml" {
  metadata {
    name = "clearml"
  }
}


resource "kubernetes_ingress_v1" "ingress" {
  metadata {
    name = "clearml-ingress"
    namespace = "clearml"
  }
  spec {
    ingress_class_name = "nginx"
    rule {
      host = "clearml.voxelplatform.com"
      http {
        path {
          backend {
            service {
              name = "clearml-webserver"
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
      host = "app.clearml.voxelplatform.com"
      http {
        path {
          backend {
            service {
              name = "clearml-webserver"
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
      host = "api.clearml.voxelplatform.com"
      http {
        path {
          backend {
            service {
              name = "clearml-apiserver"
            port {
              number = 8008
            }
          }
          }
          path = "/"
        }
      }
    }
    rule {
      host = "files.clearml.voxelplatform.com"
      http {
        path {
          backend {
            service {
              name = "clearml-fileserver"
            port {
              number = 8081
            }
          }
          }
          path = "/"
        }
      }
    }
    rule {
      host = "app.clearml.private.voxelplatform.com"
      http {
        path {
          backend {
            service {
              name = "clearml-webserver"
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
      host = "api.clearml.private.voxelplatform.com"
      http {
        path {
          backend {
            service {
              name = "clearml-apiserver"
            port {
              number = 8008
            }
          }
          }
          path = "/"
        }
      }
    }
    rule {
      host = "files.clearml.private.voxelplatform.com"
      http {
        path {
          backend {
            service {
              name = "clearml-fileserver"
            port {
              number = 8081
            }
          }
          }
          path = "/"
        }
      }
    }
}
}