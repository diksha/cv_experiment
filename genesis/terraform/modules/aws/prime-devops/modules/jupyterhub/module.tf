module "postgres" {
  source        = "../../../shared/eks-postgres"
  db_identifier = "jupyterhub"
  cluster_name  = var.eks_cluster_name
  account_id    = var.account_id
}

resource "helm_release" "main" {
  name             = "jupyterhub"
  namespace        = "jupyterhub"
  chart            = "jupyterhub"
  repository       = "https://jupyterhub.github.io/helm-chart/"
  version          = "2.0.0"
  wait             = true
  timeout          = 300
  create_namespace = true
  values = [templatefile("${path.module}/files/service-helm-values.yaml", {
    GOOGLE_CLIENT_ID = var.google_client_id
    })
  ]
  set_sensitive {
    name = "hub.config.GoogleOAuthenticator.client_secret"
    value = var.google_client_secret
  }
  set_sensitive {
    name  = "hub.db.url"
    value = "postgresql+psycopg2://${module.postgres.db_instance_username}:${module.postgres.db_instance_password}@${module.postgres.db_instance_host}:${module.postgres.db_instance_port}/${module.postgres.db_instance_name}"
  }
}