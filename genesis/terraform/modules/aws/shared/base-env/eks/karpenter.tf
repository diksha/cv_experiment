module "cluster_autoscaler" {
  source                           = "lablabs/eks-cluster-autoscaler/aws"
  version                          = "1.6.1"
  cluster_name                     = module.eks.cluster_name
  cluster_identity_oidc_issuer     = module.eks.oidc_provider
  cluster_identity_oidc_issuer_arn = module.eks.oidc_provider_arn
}
