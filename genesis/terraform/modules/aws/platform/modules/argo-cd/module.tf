resource "random_password" "argo_password" {
  length           = 20
  special          = true
  override_special = "_-@"
}

resource "helm_release" "argo_cd" {
  name             = "argo-cd"
  namespace        = kubernetes_namespace.argo_cd.metadata[0].name
  chart            = "argo-cd"
  repository       = "https://argoproj.github.io/argo-helm"
  version          = "4.5.3"
  wait             = true
  timeout          = 300
  create_namespace = false
  values = [templatefile("${path.module}/files/service-helm-values.yaml", {
    # CERTIFICATE_ARN      = var.certificate_arn
    DOMAIN               = var.domain
    })
  ]
  set_sensitive {
    name = "notifications.notifiers.service\\.slack"
    value = yamlencode({
      token = var.argo_slack_token
    })
  }
  set_sensitive {
    name = "server.config.dex\\.config"
    value = yamlencode({
      connectors = [
        {
          config = {
            issuer                 = "https://accounts.google.com"
            clientID               = var.google_client_id
            clientSecret           = var.google_client_secret
            serviceAccountFilePath = "/extra-secrets/google-sa.json"
            adminEmail             = "kacie@voxelai.com"
            redirectURI            = "https://${var.domain}/api/dex/callback"
          }
          type = "google"
          id   = "google"
          name = "Google"
        }
      ]
    })
  }

  set_sensitive {
    name = "configs.secret.argocdServerAdminPassword"
    value = random_password.argo_password.bcrypt_hash
  }
}

resource "kubernetes_secret" "extra_secrets" {
  metadata {
    name      = "extra-secrets"
    namespace = kubernetes_namespace.argo_cd.metadata[0].name
  }
  binary_data = {
    "google-sa.json" = var.google_groups_sa_json_base64_encoded
  }
}

resource "kubernetes_namespace" "argo_cd" {
  metadata {
    name = "argo-cd"
  }
}

resource "kubernetes_namespace" "argo_state_production" {
  metadata {
    name = "argo-state-production"
  }
}