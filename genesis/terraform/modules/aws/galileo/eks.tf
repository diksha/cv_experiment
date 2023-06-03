data "aws_iam_roles" "roles" {
  name_regex  = ".*AdministratorAccess.*"
  path_prefix = "/aws-reserved/sso.amazonaws.com/"
}

data "aws_iam_roles" "terraform_roles" {
  name_regex  = ".*TerraformAccess.*"
  path_prefix = "/"
}

locals {
  sso_role_arns = [
    for parts in [for arn in data.aws_iam_roles.roles.arns : split("/", arn)] :
    {
      rolearn  = format("%s/%s", parts[0], element(parts, length(parts) - 1))
      username = parts[length(parts) - 1]
      groups   = ["system:masters"]
    }
  ]
  terraform_role_arns = [
    for parts in [for arn in data.aws_iam_roles.terraform_roles.arns : split("/", arn)] :
    {
      rolearn  = format("%s/%s", parts[0], element(parts, length(parts) - 1))
      username = parts[length(parts) - 1]
      groups   = ["system:masters"]
    }
  ]
}

module "eks_galileo" {
  source                               = "terraform-aws-modules/eks/aws"
  version                              = "19.10.0"
  cluster_name                         = var.cluster_name
  cluster_version                      = var.cluster_version
  cluster_endpoint_private_access      = var.cluster_endpoint_private_access
  cluster_endpoint_public_access       = var.cluster_endpoint_public_access
  cluster_endpoint_public_access_cidrs = var.cluster_endpoint_public_access_cidrs
  cluster_enabled_log_types            = var.cluster_enabled_log_types
  cluster_addons = {
    vpc-cni            = {}
    aws-ebs-csi-driver = {}
  }
  create_kms_key            = var.create_kms_key
  cluster_encryption_config = var.cluster_encryption_config
  enable_kms_key_rotation   = var.enable_kms_key_rotation
  vpc_id                    = module.galileo_vpc.vpc_id
  subnet_ids                = module.galileo_vpc.private_subnet_ids
  eks_managed_node_groups = {
    galileo_core = {
      ami_type       = "AL2_x86_64"
      instance_types = ["m5a.large"]
      name           = "galileo-core"
      disk_size      = 200
      min_size       = 4
      max_size       = 6
      desired_size   = 5
      block_device_mappings = [
        {
          device_name = "/dev/xvda"
          ebs = {
            volume_size           = 200
            volume_type           = "gp2"
            encrypted             = true
            delete_on_termination = true
          }
        },
      ]
      labels = {
        galileo-node-type = "galileo-core"
        "nvidia.com/gpu" = "false"
      }
      launch_template_tags = {
        "map-migrated" = "d-server-00swbp99drezfh"
      }
      tags = {
        "k8s.io/cluster-autoscaler/${var.cluster_name}" = "owned",
        "k8s.io/cluster-autoscaler/enabled"             = "true",
      }
      max_unavailable            = 1
      iam_role_attach_cni_policy = true
      iam_role_additional_policies = {
        eks_worker_node_policy = "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy",
        ecr_policy             = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly",
        autoscaler_policy      = "arn:aws:iam::${var.target_account_id}:policy/ClusterAutoscaler",
        s3_policy              = "arn:aws:iam::aws:policy/AmazonS3FullAccess",
        ssm_policy             = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore",
        cloudwatch_policy      = "arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy",
        ebs_policy             = "arn:aws:iam::aws:policy/service-role/AmazonEBSCSIDriverPolicy",
      }
    }
    galileo_runner = {
      ami_type       = "AL2_x86_64"
      instance_types = ["m5a.xlarge"]
      name           = "galileo-runner"
      disk_size      = 200
      min_size       = 1
      max_size       = 5
      desired_size   = 1
      block_device_mappings = [
        {
          device_name = "/dev/xvda"
          ebs = {
            volume_size           = 200
            volume_type           = "gp2"
            encrypted             = true
            delete_on_termination = true
          }
        },
      ]
      labels = {
        galileo-node-type = "galileo-runner"
        "nvidia.com/gpu" = "false"
      }
      launch_template_tags = {
        "map-migrated" = "d-server-00swbp99drezfh"
      }
      tags = {
        "k8s.io/cluster-autoscaler/${var.cluster_name}" = "owned",
        "k8s.io/cluster-autoscaler/enabled"             = "true",
      }
      max_unavailable            = 1
      iam_role_attach_cni_policy = true
      iam_role_additional_policies = {
        eks_worker_node_policy = "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy",
        ecr_policy             = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly",
        autoscaler_policy      = "arn:aws:iam::${var.target_account_id}:policy/ClusterAutoscaler",
        s3_policy              = "arn:aws:iam::aws:policy/AmazonS3FullAccess",
        ssm_policy             = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore",
        cloudwatch_policy      = "arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy",
        ebs_policy             = "arn:aws:iam::aws:policy/service-role/AmazonEBSCSIDriverPolicy",
      }
    }
  }
  enable_irsa               = true
  manage_aws_auth_configmap = true
  aws_auth_roles = concat([
    {
      rolearn  = aws_iam_role.galileo.arn
      username = "Galileo"
      groups   = ["system:masters"]
    },
    {
      rolearn  = "arn:aws:iam::${var.target_account_id}:role/AdministratorAccess"
      username = "administrator"
      groups   = ["system:masters"]
    }
  ], local.sso_role_arns, local.terraform_role_arns)
}
