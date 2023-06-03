moved {
  from = module.base-account-setup.module.vpc_primary
  to = module.vpc.module.vpc_primary
}

locals {
  oidc_provider = replace(data.aws_eks_cluster.cluster.identity[0].oidc[0].issuer, "https://", "")
  vpc_name      = "devops"
  cluster_name  = "jenkins"
}

module "vpc" {
  providers = {
    aws = aws
  }
  source             = "../shared/vpc-and-subnets"
  target_account_id  = var.target_account_id
  environment        = var.environment
  vpc_cidr_root      = var.devops_vpc_cidr_root
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
  source                                = "../shared/base-env"
  account_id                            = var.target_account_id
  primary_region                        = var.primary_region
  environment                           = var.environment
  eks_should_create_standard_node_group = true
  eks_should_create_gpu_node_group      = true
  eks_cpu_instance_types                = ["t3.large", "t3.xlarge"]
  eks_gpu_instance_types                = ["g4dn.xlarge"]
  eks_default_max_instance_count        = 50
  vpc_cidr_root                         = var.devops_vpc_cidr_root
  vpc_name                              = local.vpc_name
  vpc_id                                = module.vpc.vpc_id
  private_subnets                       = module.vpc.private_subnet_ids
  security_group_ids                    = module.vpc.vpc_security_group_ids
  cluster_name                          = local.cluster_name
  cluster_version                       = "1.24"
  eks_extra_node_policies               = {}
  # As we enabled proxy protocol on nginx ingress to pass for real client ip to the services
  # we need to add :PROXY to the tcp services to work with it.
  ingress_tcp_services = {
    5433 : "sematic/sematic-postgres-postgresql-ha-pgpool:5432:PROXY",
  }
  eks_k8s_auth_extra_config = [
    {
      rolearn  = "arn:aws:iam::${var.target_account_id}:role/AWSReservedSSO_DeveloperAccess_f74b7dd970403af6"
      username = "developer_access"
      groups   = ["sematic-access", "eks-console-dashboard-full-view-access-group"]
    }
  ]
  eks_k8s_group_role_bindings = {
    "sematic_access" = {
      group     = "sematic-access"
      role      = "cluster-admin"
      namespace = "sematic"
    }
  }

  eks_extra_node_groups = {
    gpu_node_group_medium = {
      capacity_type  = "ON_DEMAND"
      instance_types = ["g4dn.2xlarge"]
      ami_type       = "AL2_x86_64_GPU"
      disk_size      = 200
      block_device_mappings = [
        {
          device_name = "/dev/xvda"
          ebs = {
            volume_size           = 200
            volume_type           = "gp3"
            iops                  = 3000
            encrypted             = true
            delete_on_termination = true
          }
        },
      ]
      taints = [{
        key    = "nvidia.com/gpu"
        value  = "true"
        effect = "NO_SCHEDULE"
      }]
      tags = {
        "nvidia.com/gpu"                                                          = "true"
        "k8s.io/cluster-autoscaler/enabled"                                       = "true"
        "k8s.io/cluster-autoscaler/node-template/label/nvidia.com/gpu"            = "true"
        "k8s.io/cluster-autoscaler/node-template/taint/dedicated: nvidia.com/gpu" = "true:NoSchedule"
      }
      launch_template_tags = {
        "nvidia.com/gpu"                                                          = "true"
        "map-migrated"                                                            = "d-server-00swbp99drezfh"
        "k8s.io/cluster-autoscaler/node-template/label/nvidia.com/gpu"            = "true"
        "k8s.io/cluster-autoscaler/node-template/taint/dedicated: nvidia.com/gpu" = "true:NoSchedule"
      }
      labels = {
        "nvidia.com/gpu" = "true"
      }
      min_size = 0
      max_size = 10
      iam_role_additional_policies = {
        prime_devops_gpu_large_kinesis_read_only = "arn:aws:iam::aws:policy/AmazonKinesisVideoStreamsReadOnlyAccess"
        prime_devops_ssm_access                  = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
      }
    }
    gpu_node_group_large = {
      capacity_type  = "ON_DEMAND"
      instance_types = ["g4dn.4xlarge"]
      ami_type       = "AL2_x86_64_GPU"
      disk_size      = 200
      block_device_mappings = {
        xvda = {
          device_name = "/dev/xvda"
          ebs = {
            volume_size           = 200
            volume_type           = "gp3"
            iops                  = 3000
            encrypted             = true
            delete_on_termination = true
          }
        }
      }
      taints = [{
        key    = "nvidia.com/gpu"
        value  = "true"
        effect = "NO_SCHEDULE"
      }]
      tags = {
        "nvidia.com/gpu"                                                          = "true"
        "k8s.io/cluster-autoscaler/enabled"                                       = "true"
        "k8s.io/cluster-autoscaler/node-template/label/nvidia.com/gpu"            = "true"
        "k8s.io/cluster-autoscaler/node-template/taint/dedicated: nvidia.com/gpu" = "true:NoSchedule"
      }
      launch_template_tags = {
        "nvidia.com/gpu"                                                          = "true"
        "map-migrated"                                                            = "d-server-00swbp99drezfh"
        "k8s.io/cluster-autoscaler/node-template/label/nvidia.com/gpu"            = "true"
        "k8s.io/cluster-autoscaler/node-template/taint/dedicated: nvidia.com/gpu" = "true:NoSchedule"
      }
      labels = {
        "nvidia.com/gpu" = "true"
      }
      min_size = 0
      max_size = 10
      iam_role_additional_policies = {
        prime_devops_kvs_readonly = "arn:aws:iam::aws:policy/AmazonKinesisVideoStreamsReadOnlyAccess"
        prime_devops_ssm_access   = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
      }
    }
    g5_node_group_small = {
      capacity_type  = "ON_DEMAND"
      instance_types = ["g5.xlarge"]
      ami_type       = "AL2_x86_64_GPU"
      disk_size      = 200
      block_device_mappings = [
        {
          device_name = "/dev/xvda"
          ebs = {
            volume_size           = 200
            volume_type           = "gp3"
            iops                  = 3000
            encrypted             = true
            delete_on_termination = true
          }
        },
      ]
      taints = [{
        key    = "nvidia.com/gpu"
        value  = "true"
        effect = "NO_SCHEDULE"
      }]
      tags = {
        "nvidia.com/gpu"                                                          = "true"
        "k8s.io/cluster-autoscaler/enabled"                                       = "true"
        "k8s.io/cluster-autoscaler/node-template/label/nvidia.com/gpu"            = "true"
        "k8s.io/cluster-autoscaler/node-template/taint/dedicated: nvidia.com/gpu" = "true:NoSchedule"
      }
      launch_template_tags = {
        "nvidia.com/gpu"                                                          = "true"
        "map-migrated"                                                            = "d-server-00swbp99drezfh"
        "k8s.io/cluster-autoscaler/node-template/label/nvidia.com/gpu"            = "true"
        "k8s.io/cluster-autoscaler/node-template/taint/dedicated: nvidia.com/gpu" = "true:NoSchedule"
      }
      labels = {
        "nvidia.com/gpu" = "true"
      }
      min_size = 0
      max_size = 10
      iam_role_additional_policies = {
        prime_devops_gpu_large_kinesis_read_only = "arn:aws:iam::aws:policy/AmazonKinesisVideoStreamsReadOnlyAccess"
        prime_devops_ssm_access                  = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
      }
    }
    g5_node_group_medium = {
      capacity_type  = "ON_DEMAND"
      instance_types = ["g5.2xlarge"]
      ami_type       = "AL2_x86_64_GPU"
      disk_size      = 200
      block_device_mappings = [
        {
          device_name = "/dev/xvda"
          ebs = {
            volume_size           = 200
            volume_type           = "gp3"
            iops                  = 3000
            encrypted             = true
            delete_on_termination = true
          }
        },
      ]
      taints = [{
        key    = "nvidia.com/gpu"
        value  = "true"
        effect = "NO_SCHEDULE"
      }]
      tags = {
        "nvidia.com/gpu"                                                          = "true"
        "k8s.io/cluster-autoscaler/enabled"                                       = "true"
        "k8s.io/cluster-autoscaler/node-template/label/nvidia.com/gpu"            = "true"
        "k8s.io/cluster-autoscaler/node-template/taint/dedicated: nvidia.com/gpu" = "true:NoSchedule"
      }
      launch_template_tags = {
        "nvidia.com/gpu"                                                          = "true"
        "map-migrated"                                                            = "d-server-00swbp99drezfh"
        "k8s.io/cluster-autoscaler/node-template/label/nvidia.com/gpu"            = "true"
        "k8s.io/cluster-autoscaler/node-template/taint/dedicated: nvidia.com/gpu" = "true:NoSchedule"
      }
      labels = {
        "nvidia.com/gpu" = "true"
      }
      min_size = 0
      max_size = 10
      iam_role_additional_policies = {
        prime_devops_gpu_large_kinesis_read_only = "arn:aws:iam::aws:policy/AmazonKinesisVideoStreamsReadOnlyAccess"
        prime_devops_ssm_access                  = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
      }
    }
    g5_node_group_large = {
      capacity_type  = "ON_DEMAND"
      instance_types = ["g5.4xlarge"]
      ami_type       = "AL2_x86_64_GPU"
      disk_size      = 200
      block_device_mappings = {
        xvda = {
          device_name = "/dev/xvda"
          ebs = {
            volume_size           = 200
            volume_type           = "gp3"
            iops                  = 3000
            encrypted             = true
            delete_on_termination = true
          }
        }
      }
      taints = [{
        key    = "nvidia.com/gpu"
        value  = "true"
        effect = "NO_SCHEDULE"
      }]
      tags = {
        "nvidia.com/gpu"                                                          = "true"
        "k8s.io/cluster-autoscaler/enabled"                                       = "true"
        "k8s.io/cluster-autoscaler/node-template/label/nvidia.com/gpu"            = "true"
        "k8s.io/cluster-autoscaler/node-template/taint/dedicated: nvidia.com/gpu" = "true:NoSchedule"
      }
      launch_template_tags = {
        "nvidia.com/gpu"                                                          = "true"
        "map-migrated"                                                            = "d-server-00swbp99drezfh"
        "k8s.io/cluster-autoscaler/node-template/label/nvidia.com/gpu"            = "true"
        "k8s.io/cluster-autoscaler/node-template/taint/dedicated: nvidia.com/gpu" = "true:NoSchedule"
      }
      labels = {
        "nvidia.com/gpu" = "true"
      }
      min_size = 0
      max_size = 10
      iam_role_additional_policies = {
        prime_devops_kvs_readonly = "arn:aws:iam::aws:policy/AmazonKinesisVideoStreamsReadOnlyAccess"
        prime_devops_ssm_access   = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
      }
    }
    cpu_node_group_medium = {
      capacity_type  = "ON_DEMAND"
      instance_types = ["m5.2xlarge"]
      ami_type       = "AL2_x86_64"
      launch_template_tags = {
        "nvidia.com/gpu"                                                          = "false"
        "k8s.io/cluster-autoscaler/node-template/label/nvidia.com/gpu"            = "false"
        "k8s.io/cluster-autoscaler/node-template/taint/dedicated: nvidia.com/gpu" = "false:NoSchedule"
        "map-migrated"                                                            = "d-server-00swbp99drezfh"
      }
      tags = {
        "nvidia.com/gpu"                                                          = "false"
        "k8s.io/cluster-autoscaler/enabled"                                       = "true"
        "k8s.io/cluster-autoscaler/node-template/label/nvidia.com/gpu"            = "false"
        "k8s.io/cluster-autoscaler/node-template/taint/dedicated: nvidia.com/gpu" = "false:NoSchedule"
      }
      labels = {
        "nvidia.com/gpu" = "false"
      }
      min_size = 0
      max_size = 20
    }
    cpu_node_group_small = {
      capacity_type  = "ON_DEMAND"
      instance_types = ["m5.xlarge"]
      ami_type       = "AL2_x86_64"
      launch_template_tags = {
        "nvidia.com/gpu"                                                          = "false"
        "k8s.io/cluster-autoscaler/node-template/label/nvidia.com/gpu"            = "false"
        "k8s.io/cluster-autoscaler/node-template/taint/dedicated: nvidia.com/gpu" = "false:NoSchedule"
        "map-migrated"                                                            = "d-server-00swbp99drezfh"
      }
      tags = {
        "nvidia.com/gpu"                                                          = "false"
        "k8s.io/cluster-autoscaler/enabled"                                       = "true"
        "k8s.io/cluster-autoscaler/node-template/label/nvidia.com/gpu"            = "false"
        "k8s.io/cluster-autoscaler/node-template/taint/dedicated: nvidia.com/gpu" = "false:NoSchedule"
      }
      labels = {
        "nvidia.com/gpu" = "false"
      }
      min_size = 0
      max_size = 20
    }
  }
}


module "devops_observability" {
  providers = {
    aws = aws
  }
  source                   = "../shared/eks-observability"
  account_id               = var.target_account_id
  cluster_name             = module.base-account-setup.eks_cluster_name
  aws_region               = var.primary_region
  observability_identifier = "${var.observability_identifier}-devops"
  grafana_url              = var.grafana_url
  grafana_api_key          = var.grafana_api_key
  grafana_irsa_arn         = var.grafana_irsa_arn
  register_with_grafana    = true
}

resource "aws_vpc_endpoint" "s3_gateway_primary_region" {
  vpc_id       = module.vpc.vpc_id
  service_name = "com.amazonaws.${var.primary_region}.s3"
  route_table_ids = concat(
    tolist(module.vpc.private_route_table_ids),
    tolist(module.vpc.public_route_table_ids),
  )
}


module "jupyterhub" {
  providers = {
    aws = aws
  }
  source               = "./modules/jupyterhub"
  account_id           = var.target_account_id
  eks_cluster_name     = module.base-account-setup.eks_cluster_name
  google_client_id     = var.google_client_id
  google_client_secret = var.google_client_secret
}

module "flink_operator" {
  providers = {
    aws = aws
  }
  source             = "../shared/flink-operator"
  account_id         = var.target_account_id
  cluster_name       = module.base-account-setup.eks_cluster_name
  namespace_to_watch = "flink-testing"
}

module "flink_testing" {
  providers = {
    aws = aws
  }
  source         = "./modules/flink-testing"
  account_id     = var.target_account_id
  primary_region = var.primary_region
  oidc_provider  = local.oidc_provider
}
