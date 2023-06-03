module "argo-production-perception-registration" {
  providers = {
    aws = aws.production
  }
  source         = "../shared/argo-cluster-registration"
  cluster_name   = "perception"
  cluster_alias  = "production-perception"
  admin_username = var.argo_admin_username
  admin_password = var.argo_admin_password
  server_domain  = var.argo_domain
}

module "argo-production-product-registration" {
  providers = {
    aws = aws.production
  }
  source         = "../shared/argo-cluster-registration"
  cluster_name   = "product"
  cluster_alias  = "production-product"
  admin_username = var.argo_admin_username
  admin_password = var.argo_admin_password
  server_domain  = var.argo_domain
}

module "argo-staging-product-registration" {
  providers = {
    aws = aws.staging
  }
  source         = "../shared/argo-cluster-registration"
  cluster_name   = "product"
  cluster_alias  = "staging-product"
  admin_username = var.argo_admin_username
  admin_password = var.argo_admin_password
  server_domain  = var.argo_domain
}
