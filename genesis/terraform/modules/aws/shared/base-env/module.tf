module "eks_services_cluster" {
  providers = {
    aws = aws
  }
  source                                  = "./eks"
  cluster_name                            = var.cluster_name
  cluster_version                         = var.cluster_version
  vpc_id                                  = var.vpc_id
  vpc_name                                = var.vpc_name
  subnet_ids                              = var.private_subnets
  security_group_ids                      = var.security_group_ids
  environment                             = var.environment
  account_id                              = var.account_id
  gpu_instance_types                      = var.eks_gpu_instance_types
  cpu_instance_types                      = var.eks_cpu_instance_types
  should_create_gpu_node_group            = var.eks_should_create_gpu_node_group
  should_create_standard_node_group       = var.eks_should_create_standard_node_group
  cpu_instance_max_size                   = var.eks_default_max_instance_count
  gpu_instance_max_size                   = var.eks_default_max_instance_count
  extra_node_policies                     = var.eks_extra_node_policies
  k8s_auth_extra_config                   = var.eks_k8s_auth_extra_config
  k8s_group_role_bindings                 = var.eks_k8s_group_role_bindings
  default_disk_size_gb                    = var.eks_default_disk_size
  extra_node_groups                       = var.eks_extra_node_groups
  create_alarm                            = var.create_alarm
  aws_notifications_slack_app_webhook_url = var.aws_notifications_slack_app_webhook_url
}

data "aws_subnet" "subnets" {
  for_each   = toset(["10.${var.vpc_cidr_root}.0.0/20", "10.${var.vpc_cidr_root}.128.0/20"])
  cidr_block = each.key
  depends_on = [
    module.eks_services_cluster
  ]
}

module "ingress_nginx" {
  enable_nginx_ingress      = var.enable_nginx_ingress
  source                    = "../ingress-nginx"
  cluster_name              = module.eks_services_cluster.cluster_name
  account_id                = var.account_id
  static_ipv4s              = "10.${var.vpc_cidr_root}.0.100,10.${var.vpc_cidr_root}.128.100"
  subnet_ids                = join(",", [for subnet in data.aws_subnet.subnets : subnet.id])
  tcp_services              = var.ingress_tcp_services
  ssl_ports                 = var.ingress_ssl_ports
  subject_alternative_names = var.ingress_cert_subject_alternative_names
}
