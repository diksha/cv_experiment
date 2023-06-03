module "cloudflared" {
  providers = {
    aws = aws
  }
  source         = "../shared/cloudflare-tunnel"
  cluster_name   = module.eks_galileo.cluster_name
  primary_region = var.primary_region
  tunnel_token   = var.cloudflare_tunnel_token
}
