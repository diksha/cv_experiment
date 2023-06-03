moved {
  from = module.base-account-setup.module.vpc_primary
  to = module.vpc.module.vpc_primary
}

locals {
  oidc_provider = replace(data.aws_eks_cluster.cluster.identity[0].oidc[0].issuer, "https://", "")
  vpc_name      = "primary"
  cluster_name  = "services-cluster"
  environment   = "production"
}


module "vpc" {
  providers = {
    aws = aws
  }
  source             = "../shared/vpc-and-subnets"
  target_account_id  = var.target_account_id
  environment        = local.environment
  vpc_cidr_root      = var.vpc_cidr_root
  vpc_name           = local.vpc_name
  enable_nat_gateway = true
  public_subnet_tags = {
    "kubernetes.io/role/elb"                      = "1"
    "kubernetes.io/cluster/${local.cluster_name}" = "shared"
  }
  private_subnet_tags = {
    "kubernetes.io/role/internal-elb"             = "1"
    "kubernetes.io/cluster/${local.cluster_name}" = "shared"
  }
}


module "base-account-setup" {
  providers = {
    aws = aws
  }
  source                 = "../shared/base-env"
  account_id             = var.target_account_id
  primary_region         = var.primary_region
  environment            = local.environment
  dev_mode               = false
  vpc_cidr_root          = var.vpc_cidr_root
  vpc_name               = local.vpc_name
  vpc_id                 = module.vpc.vpc_id
  private_subnets        = module.vpc.private_subnet_ids
  security_group_ids     = module.vpc.vpc_security_group_ids
  cluster_name           = local.cluster_name
  cluster_version        = "1.24"
  eks_cpu_instance_types = ["t3.2xlarge"]
  eks_gpu_instance_types = ["g4dn.xlarge", "g4dn.4xlarge"]
  enable_nginx_ingress   = true
  # As we enabled proxy protocol on nginx ingress to pass for real client ip to the services
  # we need to add :PROXY to the tcp services to work with it.
  ingress_tcp_services = {
    7687 : "metaverse/metaverse-neo4j:7687:PROXY",
    7688 : "metaverse-internal/metaverse-internal-neo4j:7687:PROXY",
    15119 : "excalidraw/excalidraw-collab-generic:80:PROXY",
  }
  ingress_ssl_ports                      = "443,7687,7688,15119"
  ingress_cert_subject_alternative_names = ["*.clearml.private.voxelplatform.com"]
}

resource "aws_vpc_endpoint" "s3_gateway_primary_region" {
  vpc_id       = module.vpc.vpc_id
  service_name = "com.amazonaws.${var.primary_region}.s3"
  route_table_ids = concat(
    tolist(module.vpc.private_route_table_ids),
    tolist(module.vpc.public_route_table_ids),
  )
}

module "argo-cd" {
  providers = {
    aws = aws
  }
  source                               = "./modules/argo-cd"
  account_id                           = var.target_account_id
  domain                               = "argo.${var.root_domain}"
  cluster_name                         = module.base-account-setup.eks_cluster_name
  google_client_id                     = var.google_client_id
  google_client_secret                 = var.google_client_secret
  argo_slack_token                     = var.slack_token
  google_groups_sa_json_base64_encoded = var.google_groups_sa_json_base64_encoded
}


module "sentry" {
  providers = {
    aws = aws
  }
  source               = "./modules/sentry"
  account_id           = var.target_account_id
  cluster_name         = module.base-account-setup.eks_cluster_name
  google_client_id     = var.google_client_id
  google_client_secret = var.google_client_secret
  slack                = var.sentry_slack
}

module "clearml" {
  providers = {
    aws = aws
  }
  source       = "./modules/clearml"
  account_id   = var.target_account_id
  cluster_name = module.base-account-setup.eks_cluster_name
}


module "grafana" {
  providers = {
    aws = aws
  }
  source                               = "./modules/eks-grafana"
  account_id                           = var.target_account_id
  domain                               = "grafana.${var.root_domain}"
  cluster_name                         = module.base-account-setup.eks_cluster_name
  google_client_id                     = var.google_client_id
  google_client_secret                 = var.google_client_secret
  slack_token                          = var.slack_token
  google_groups_sa_json_base64_encoded = var.google_groups_sa_json_base64_encoded
  oidc_provider                        = local.oidc_provider
}

module "observability" {
  source                             = "../shared/eks-observability"
  account_id                         = var.target_account_id
  cluster_name                       = module.base-account-setup.eks_cluster_name
  force_deploy_dashboard_config_maps = true # This should only be done once as the dashboards allow for configurable data source which is reusable in other clusters
  grafana_url                        = "https://grafana.${var.root_domain}"
  grafana_api_key                    = module.grafana.grafana_api_key
  grafana_irsa_arn                   = module.grafana.grafana_irsa_arn
  observability_identifier           = var.observability_identifier
  register_with_grafana              = true
  aws_region                         = var.primary_region
}

module "hoppscotch" {
  providers = {
    aws = aws
  }
  source         = "./modules/hoppscotch"
  account_id     = var.target_account_id
  domain         = "api-dev.${var.root_domain}"
  cluster_name   = module.base-account-setup.eks_cluster_name
  primary_region = var.primary_region
}

module "cloudflared" {
  providers = {
    aws = aws
  }
  source         = "../shared/cloudflare-tunnel"
  cluster_name   = module.base-account-setup.eks_cluster_name
  primary_region = var.primary_region
  tunnel_token   = var.cloudflare_tunnel_token
}

module "retool" {
  providers = {
    aws = aws
  }
  source               = "./modules/retool"
  account_id           = var.target_account_id
  cluster_name         = module.base-account-setup.eks_cluster_name
  google_client_id     = var.google_client_id
  google_client_secret = var.google_client_secret
  retool_license_key   = var.retool_license_key
  eks_cluster_name     = module.base-account-setup.eks_cluster_name
}

module "posthog" {
  providers = {
    aws = aws
  }
  source               = "./modules/posthog"
  account_id           = var.target_account_id
  cluster_name         = module.base-account-setup.eks_cluster_name
  google_client_id     = var.google_client_id
  google_client_secret = var.google_client_secret
}

module "metaverse" {
  providers = {
    aws = aws
  }
  source        = "./modules/metaverse"
  account_id    = var.target_account_id
  cluster_name  = module.base-account-setup.eks_cluster_name
  db_identifier = "metaverse"
  environment   = "production"
}

module "metaverse_internal" {
  providers = {
    aws = aws
  }
  source        = "./modules/metaverse"
  account_id    = var.target_account_id
  cluster_name  = module.base-account-setup.eks_cluster_name
  db_identifier = "metaverse"
  environment   = "internal"
}

module "excalidraw" {
  providers = {
    aws = aws
  }
  source         = "./modules/excalidraw"
  account_id     = var.target_account_id
  cluster_name   = module.base-account-setup.eks_cluster_name
  primary_region = var.primary_region
}

module "snipeit" {
  providers = {
    aws = aws
  }
  source       = "./modules/snipeit"
  account_id   = var.target_account_id
  cluster_name = module.base-account-setup.eks_cluster_name
  snipeit_key  = var.snipeit_key
}

module "privatebin" {
  providers = {
    aws = aws
  }
  source       = "./modules/privatebin"
  account_id   = var.target_account_id
  cluster_name = module.base-account-setup.eks_cluster_name
}


module "atlantis" {
  providers = {
    aws = aws
  }
  source                                  = "./modules/atlantis"
  account_id                              = var.target_account_id
  cluster_name                            = module.base-account-setup.eks_cluster_name
  google_sa_json_terraform_base64_encoded = var.google_sa_json_terraform_base64_encoded
  atlantis                                = var.atlantis
  oidc_provider                           = local.oidc_provider
}


module "foxglove" {
  providers = {
    aws = aws
  }
  source       = "./modules/foxglove"
  account_id   = var.target_account_id
  cluster_name = module.base-account-setup.eks_cluster_name
  root_domain  = var.root_domain
}
