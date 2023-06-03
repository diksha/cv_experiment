locals {
  node_policies = merge({
    base_env_ssm_policy = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
  }, var.extra_node_policies)
}


module "eks" {
  providers = {
    aws = aws
  }
  source                                 = "terraform-aws-modules/eks/aws"
  version                                = "19.10.0"
  cluster_name                           = var.cluster_name
  cluster_version                        = var.cluster_version
  vpc_id                                 = var.vpc_id
  subnet_ids                             = var.subnet_ids
  cloudwatch_log_group_retention_in_days = 7
  // Security
  enable_irsa                          = true
  cluster_endpoint_public_access       = true
  cluster_endpoint_public_access_cidrs = ["0.0.0.0/0"]
  cluster_endpoint_private_access      = true
  cluster_enabled_log_types	           = ["audit", "api", "authenticator", "controllerManager", "scheduler"]
  create_aws_auth_configmap            = true
  manage_aws_auth_configmap            = true
  create_kms_key                       = false
  cluster_encryption_config            = {}
  aws_auth_roles = concat(
    [
      {
        rolearn  = "arn:aws:iam::${var.account_id}:role/AdministratorAccess"
        username = "administrator"
        groups   = ["system:masters"]
      }
    ], concat(local.sso_role_arns, local.terraform_role_arns, var.k8s_auth_extra_config)
  )
  cluster_addons = {
    coredns = {
      resolve_conflicts = "OVERWRITE"
    }
    kube-proxy = {
      resolve_conflicts = "OVERWRITE"
    }
    vpc-cni = {
      resolve_conflicts = "OVERWRITE"
    }
  }

  cluster_security_group_additional_rules = {
    ingress_https_from_devbox_for_kubectl = {
      description = "Ingress https from devbox"
      protocol    = "tcp"
      from_port   = 0
      to_port     = 443
      type        = "ingress"
      cidr_blocks = ["0.0.0.0/0"]
    }
  }

  node_security_group_additional_rules = {
    ingress_self_coredns_tcp = {
      description = "Node to node CoreDNS"
      protocol    = "tcp"
      from_port   = 53
      to_port     = 53
      type        = "ingress"
      self        = true
    }
    # https://github.com/kubernetes-sigs/metrics-server/issues/1024#issuecomment-1129914389
    ingress_metrics_server_allow_access_from_control_plane = {
      type                          = "ingress"
      protocol                      = "-1"
      from_port                     = 0
      to_port                       = 4443
      source_cluster_security_group = true
      description                   = "Cluster API to Node group for Metrics Server"
    }

    ingress_self_all = {
      description = "Node to node all ports/protocols"
      protocol    = "-1"
      from_port   = 0
      to_port     = 0
      type        = "ingress"
      self        = true
    }
  }

  node_security_group_tags = {
    "karpenter.sh/discovery/${var.cluster_name}" = var.cluster_name
  }

  create_iam_role = true
  eks_managed_node_group_defaults = {
    ami_type                     = "AL2_x86_64"
    instance_types               = var.instance_types
    vpc_security_group_ids       = var.security_group_ids
    iam_role_attach_cni_policy   = true
    iam_role_additional_policies = local.node_policies
    enable_monitoring            = false
    disk_size                    = var.default_disk_size_gb
    block_device_mappings = [
      {
        device_name = "/dev/xvda"
        ebs = {
          volume_size = var.default_disk_size_gb
          volume_type = "gp3"
          iops        = 3000
          # throughput            = 150
          encrypted             = true
          delete_on_termination = true
        }
      },
    ]
    iam_role_additional_policies = local.node_policies
    iam_role_attach_cni_policy   = true
    create_security_group = false
  }

  eks_managed_node_groups = merge({
    standard_node_group = {
      create         = var.should_create_standard_node_group
      capacity_type  = "ON_DEMAND"
      instance_types = var.cpu_instance_types
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
      min_size = var.cpu_instance_min_size
      max_size = var.cpu_instance_max_size
    }
    gpu_node_group = {
      create         = var.should_create_gpu_node_group
      capacity_type  = "ON_DEMAND"
      instance_types = var.gpu_instance_types
      ami_type       = "AL2_x86_64_GPU"
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
        "k8s.io/cluster-autoscaler/node-template/label/nvidia.com/gpu"            = "true"
        "k8s.io/cluster-autoscaler/node-template/taint/dedicated: nvidia.com/gpu" = "true:NoSchedule"
        "map-migrated"                                                            = "d-server-00swbp99drezfh"
      }
      labels = {
        "nvidia.com/gpu" = "true"
      }

      min_size = var.gpu_instance_min_size
      max_size = var.gpu_instance_max_size
      iam_role_additional_policies = merge({
        base_env_kinesis_video_streams_read_only = "arn:aws:iam::aws:policy/AmazonKinesisVideoStreamsReadOnlyAccess"
      }, local.node_policies)
    }
  }, var.extra_node_groups)

  fargate_profiles = {
    default = {
      name = var.cluster_name
      selectors = [
        {
          namespace = "kube-system"
          labels = {
            k8s-app = "kube-dns"
          }
        },
        {
          namespace = "fargate"
        }
      ]
      timeouts = {
        create = "20m"
        delete = "20m"
      }
    }
  }
}
