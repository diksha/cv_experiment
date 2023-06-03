locals {
  buildkite_ami_gpu_name = "voxel-buildkite-ami-gpu-v1"
  # buildkite_gpu_ami_id   = "ami-0fd5b5eca171645f8"
  buildkite_gpu_ami_id = data.aws_ami.buildkite_gpu_v1.id
  common_buildkite_stack_params = {
    AssociatePublicIpAddress           = false
    BuildkiteAgentExperiments          = "opentelemetry-tracing"
    BuildkiteAgentRelease              = "stable"
    BuildkiteAgentTimestampLines       = false
    BuildkiteAgentToken                = var.buildkite_token
    RootVolumeType                     = "gp3"
    ManagedPolicyARN                   = join(",", [
      "arn:aws:iam::aws:policy/AmazonKinesisVideoStreamsReadOnlyAccess",
      aws_iam_policy.buildkite_elastic_ci_stack_assume_role.arn,
      aws_iam_policy.buildkite_bazel_remote_cache_bucket_access.arn,
    ])
    Subnets                            = join(",", module.vpc.private_subnet_ids)
    VpcId                              = module.vpc.vpc_id
    BuildkiteTerminateInstanceAfterJob = false
    CostAllocationTagName              = "CreatedBy"
    CostAllocationTagValue             = "buildkite-elastic-ci-stack-for-aws"
    ECRAccessPolicy                    = "readonly"
    EnableAgentGitMirrorsExperiment    = false
    EnableCostAllocationTags           = true
    EnableDockerExperimental           = true
    EnableDockerLoginPlugin            = true
    EnableDockerUserNamespaceRemap     = true
    EnableECRPlugin                    = true
    EnableInstanceStorage              = false
    EnableSecretsPlugin                = true
    IMDSv2Tokens                       = "optional"
    InstanceOperatingSystem            = "linux"
    SecretsBucket                      = aws_s3_bucket.secrets.id
    ArtifactsBucket                    = aws_s3_bucket.artifacts.id
    BootstrapScriptUrl                 = "s3://${aws_s3_bucket.secrets.id}/${aws_s3_bucket_object.buildkite_bootstrap_script.key}"
  }
}

module "buildkite_bazel_remote_cache_bucket" {
  source = "../shared/s3-bucket"
  target_account_id = var.target_account_id
  primary_region = var.primary_region
  bucket_name = "voxel-bazel-remote"
}

data "aws_iam_policy_document" "buildkite_bazel_remote_cache_bucket_access" {
  statement {
    sid = "BazelRemoteCacheBucketAccess"
    actions = ["s3:*Object"]
    resources = [
      "${module.buildkite_bazel_remote_cache_bucket.bucket_arn}/*",
    ]
  }
}

resource "aws_iam_policy" "buildkite_bazel_remote_cache_bucket_access" {
  name = "BuildkiteBazelRemoteCacheBucketAccess"
  policy = data.aws_iam_policy_document.buildkite_bazel_remote_cache_bucket_access.json
}

resource "aws_s3_bucket_object" "buildkite_bootstrap_script" {
  bucket = aws_s3_bucket.secrets.id
  key = "voxel_bootstrap.sh"
  content = templatefile("${path.module}/files/buildkite-install-bazel-remote.sh", {
    AWS_REGION = var.primary_region,
    CACHE_BUCKET = module.buildkite_bazel_remote_cache_bucket.bucket_id,
  })
}

resource "aws_s3_bucket" "secrets" {
  bucket = "v-ci-cd-bk-secrets-01"
}

resource "aws_s3_bucket_acl" "secrets" {
  bucket = aws_s3_bucket.secrets.id
  acl    = "private"
}

resource "aws_s3_bucket" "artifacts" {
  bucket = "v-ci-cd-bk-artifacts-01"
}

resource "aws_s3_bucket_acl" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  acl    = "private"
}

resource "aws_s3_object" "secrets_private_ssh_key" {
  bucket         = aws_s3_bucket.secrets.id
  key            = "private_ssh_key"
  content_base64 = var.git_ssh_key_base64_encoded
  source_hash    = md5(base64decode(var.git_ssh_key_base64_encoded))
  lifecycle {
    ignore_changes = [
      version_id
    ]
  }
}

resource "aws_iam_policy" "buildkite_access" {
  name        = "BuildkiteAccess"
  description = "Buildkite Access"
  policy = jsonencode({
    Statement = [
      {
        Sid = "Buckets"
        Action = [
          "s3:ListBucketMultipartUploads",
          "s3:ListBucketVersions",
          "s3:ListBucket",
          "s3:ListMultipartUploadParts",
          "s3:GetObjectAcl",
          "s3:GetObject",
          "s3:GetBucketAcl",
          "s3:GetObjectVersionAcl",
          "s3:GetObjectVersion",
          "s3:GetStorageLensDashboard",
          "s3:GetBucketLocation",
          "s3:PutObject",
          "s3:RestoreObject",
          "s3:AbortMultipartUpload",
          "s3:DeleteObject",
        ],
        Effect = "Allow",
        Resource = [
          "arn:aws:s3:::${var.target_account_id}-sematic-ci-cd",
          "arn:aws:s3:::${var.target_account_id}-sematic-ci-cd/*",
          "arn:aws:s3:::voxel-consumable-labels/*",
          "arn:aws:s3:::voxel-consumable-labels",
          "arn:aws:s3:::voxel-datasets/*",
          "arn:aws:s3:::voxel-datasets",
          "arn:aws:s3:::voxel-lightly-input/*",
          "arn:aws:s3:::voxel-lightly-input",
          "arn:aws:s3:::voxel-lightly-output/*",
          "arn:aws:s3:::voxel-lightly-output",
          "arn:aws:s3:::voxel-logs/*",
          "arn:aws:s3:::voxel-logs",
          "arn:aws:s3:::voxel-models/*",
          "arn:aws:s3:::voxel-models",
          "arn:aws:s3:::voxel-raw-logs/*",
          "arn:aws:s3:::voxel-raw-logs",
          "arn:aws:s3:::voxel-raw-labels/*",
          "arn:aws:s3:::voxel-raw-labels",
          "arn:aws:s3:::voxel-temp/*",
          "arn:aws:s3:::voxel-temp",
          "arn:aws:s3:::voxel-users/*",
          "arn:aws:s3:::voxel-users",
          "arn:aws:s3:::voxel-storage/*",
          "arn:aws:s3:::voxel-storage",
          "arn:aws:s3:::voxel-perception/*",
          "arn:aws:s3:::voxel-perception",
          "arn:aws:s3:::voxel-portal-staging-static-resources",
          "arn:aws:s3:::voxel-portal-staging-static-resources/*",
          "arn:aws:s3:::voxel-portal-production-static-resources",
          "arn:aws:s3:::voxel-portal-production-static-resources/*",
          # voxeldata bucket is owned by Uber for labeling purposes
          "arn:aws:s3:::voxeldata/*",
          "arn:aws:s3:::voxeldata",
          # for data operations team
          "arn:aws:s3:::voxel-temp-dataops/*",
          "arn:aws:s3:::voxel-temp-dataops",
        ]
      },
      {
        Sid = "ReadWriteNoDeleteBuckets"
        Action = [
          "s3:ListBucketMultipartUploads",
          "s3:ListBucketVersions",
          "s3:ListBucket",
          "s3:ListMultipartUploadParts",
          "s3:GetObjectAcl",
          "s3:GetObject",
          "s3:GetBucketAcl",
          "s3:GetObjectVersionAcl",
          "s3:GetObjectVersion",
          "s3:GetBucketLocation",
          "s3:PutObject",
          "s3:RestoreObject",
          "s3:AbortMultipartUpload",
        ],
        Effect = "Allow",
        Resource = [
          "arn:aws:s3:::voxel-metaverse-backup/*",
          "arn:aws:s3:::voxel-metaverse-backup"
        ]
      },
      {
        Sid = "ReadOnlyBuckets"
        Action = [
          "s3:ListBucketMultipartUploads",
          "s3:ListBucketVersions",
          "s3:ListBucket",
          "s3:ListMultipartUploadParts",
          "s3:GetObjectAcl",
          "s3:GetObject",
          "s3:GetBucketAcl",
          "s3:GetObjectVersionAcl",
          "s3:GetObjectVersion",
          "s3:GetBucketLocation",
        ],
        Effect = "Allow",
        Resource = [
          "arn:aws:s3:::voxel-portal-production/*",
          "arn:aws:s3:::voxel-portal-production",
          "arn:aws:s3:::voxel-portal-staging/*",
          "arn:aws:s3:::voxel-portal-staging",
        ]
      },
      {
        Sid = "ECR"
        Action = [
          "ecr:DescribeImageScanFindings",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage",
          "ecr:CompleteLayerUpload",
          "ecr:DescribeImages",
          "ecr:ListTagsForResource",
          "ecr:UploadLayerPart",
          "ecr:ListImages",
          "ecr:InitiateLayerUpload",
          "ecr:BatchCheckLayerAvailability",
          "ecr:PutImage"
        ],
        Effect = "Allow",
        Resource = [
          # Add ECR repositories in terraform with lifecycle.
          "arn:aws:ecr:${var.primary_region}:${var.target_account_id}:repository/sematic",
          "arn:aws:ecr:${var.primary_region}:${var.target_account_id}:repository/ci-cd-base",
          "arn:aws:ecr:${var.primary_region}:${var.target_account_id}:repository/voxel-ci/ubuntu",
          "arn:aws:ecr:${var.primary_region}:${var.target_account_id}:repository/lightly/boris-250909/lightly/worker",
          # Specify repositories instead of wildcard and set resource based policies.
          "arn:aws:ecr:${var.primary_region}:360054435465:repository/*",
          "arn:aws:ecr:${var.primary_region}:115099983231:repository/*",
        ]
      },
      {
        Sid = "EC2Route53"
        Action = [
          "ec2:DetachVolume",
          "ec2:AttachVolume",
          "ec2:DeleteVolume",
          "ec2:ModifyVolume",
          "ec2:DescribeInstances",
          "ec2:TerminateInstances",
          "ec2:CreateTags",
          "route53:ChangeResourceRecordSets",
          "ec2:RunInstances",
          "ec2:ModifyInstanceAttribute",
          "ec2:StopInstances",
          "ec2:StartInstances",
          "ec2:CreateVolume"
        ],
        Effect = "Allow",
        Resource = ["*"]
      },
      {
        Sid = "ECRAuthorization"
        Action   = [
            "ecr:GetAuthorizationToken",
        ],
        Effect   = "Allow",
        Resource = "*"
      },
      {
        Sid = "SecretsManager"
        Action = [
          "secretsmanager:GetSecretValue",
          "secretsmanager:UpdateSecret",
        ],
        Effect = "Allow",
        Resource = [
          "arn:aws:secretsmanager:${var.primary_region}:*"
        ]
      },
      {
        Sid = "SecretsManagerList"
        Action   = ["secretsmanager:ListSecrets"],
        Effect   = "Allow",
        Resource = "*"
      },
      {
        Sid      = "AssumeProductionAccountRole"
        Action   = ["sts:AssumeRole"],
        Effect   = "Allow",
        Resource = ["arn:aws:iam::360054435465:role/buildkite_access_assumable_role"]
      },
      {
        Sid      = "AssumeStagingAccountRole"
        Action   = ["sts:AssumeRole"],
        Effect   = "Allow",
        Resource = ["arn:aws:iam::115099983231:role/buildkite_access_assumable_role"]
      },
      # Remove once portal moves out of root account.
      {
        Sid      = "AssumeRootAccountRole"
        Action   = ["sts:AssumeRole"],
        Effect   = "Allow",
        Resource = ["arn:aws:iam::667031391229:role/buildkite_access_assumable_role"]
      },
    ]
    Version = "2012-10-17"
  })
}

resource "aws_iam_policy" "buildkite_elastic_ci_stack_assume_role" {
  name        = "BuildkiteElasticCIStackAssumeRole"
  description = "Buildkite Elastic CI stack assume role policy"
  policy = jsonencode({
    Statement = [
      {
        Sid      = "AssumeProductionAccountRole"
        Action   = ["sts:AssumeRole"],
        Effect   = "Allow",
        Resource = ["arn:aws:iam::${var.target_account_id}:role/BuildkiteAccess"]
      },
      {
        Sid = "SecretsManager"
        Action = [
          "secretsmanager:GetSecretValue"
        ],
        Effect = "Allow",
        Resource = [
          "arn:aws:secretsmanager:${var.primary_region}:*"
        ]
      },
    ]
    Version = "2012-10-17"
  })
}


# resource "aws_s3_bucket_object" "secrets_portal_ci_cd_env" {
#   bucket         = aws_s3_bucket.secrets.id
#   key            = "env"
#   content_base64 = var.portal_ci_cd_env_base64_encoded
#   etag           = md5(var.git_ssh_key_base64_encoded)
# }


//TODO: Add 'env' file also in the bucket

resource "aws_cloudformation_stack" "buildkite_ci_cpu_4c16g_x8" {
  name = "buildkite-ci-cpu-4c16g-x8"

  parameters = merge(local.common_buildkite_stack_params, {
    AgentsPerInstance      = 8
    BuildkiteAgentTags     = "os=linux,machine-type=cpu,instance-size=m5.xlarge"
    BuildkiteQueue         = "cpu-aws-4c16g-x8"
    InstanceType           = "m5.xlarge"
    MaxSize                = 20
    MinSize                = 0
    OnDemandPercentage     = 100
    RootVolumeSize         = 400
    ScaleInIdlePeriod      = 3600
    ScaleOutFactor         = 1.0
    ScaleOutForWaitingJobs = false
  })
  capabilities = ["CAPABILITY_AUTO_EXPAND", "CAPABILITY_IAM", "CAPABILITY_NAMED_IAM"]
  template_url = "https://s3.amazonaws.com/buildkite-aws-stack/latest/aws-stack.yml"

  lifecycle {
    ignore_changes = all
  }
}

resource "aws_cloudformation_stack" "buildkite_ci_cpu_16c64g_x4" {
  name = "buildkite-ci-cpu-16c64g-x4"
  parameters = merge(local.common_buildkite_stack_params, {
    AgentsPerInstance      = 4
    BuildkiteAgentTags     = "os=linux,machine-type=cpu,instance-size=m5.4xlarge"
    BuildkiteQueue         = "cpu-aws-16c64g-x4"
    InstanceType           = "m5d.4xlarge"
    MaxSize                = 20
    MinSize                = 0
    OnDemandPercentage     = 100
    RootVolumeSize         = 200
    EnableInstanceStorage  = true
    ScaleInIdlePeriod      = 3600
    ScaleOutFactor         = 1.0
    ScaleOutForWaitingJobs = false
  })
  capabilities = ["CAPABILITY_AUTO_EXPAND", "CAPABILITY_IAM", "CAPABILITY_NAMED_IAM"]
  template_url = "https://s3.amazonaws.com/buildkite-aws-stack/latest/aws-stack.yml"
  lifecycle {
    ignore_changes = all
  }
}

resource "aws_cloudformation_stack" "buildkite_ci_gpu_16c64g_x4" {
  name = "buildkite-ci-gpu-16c64g-x4"

  parameters = merge(local.common_buildkite_stack_params, {
    AgentsPerInstance      = 4
    BuildkiteAgentTags     = "os=linux,machine-type=cpu,instance-size=g4dn.4xlarge"
    BuildkiteQueue         = "gpu-aws-16c64g-x4"
    InstanceType           = "g4dn.4xlarge"
    MaxSize                = 20
    MinSize                = 0
    OnDemandPercentage     = 100
    RootVolumeSize         = 400
    ScaleInIdlePeriod      = 3600
    ScaleOutFactor         = 1.0
    ScaleOutForWaitingJobs = false
    ImageId                = local.buildkite_gpu_ami_id
  })
  capabilities = ["CAPABILITY_AUTO_EXPAND", "CAPABILITY_IAM", "CAPABILITY_NAMED_IAM"]
  template_url = "https://s3.amazonaws.com/buildkite-aws-stack/latest/aws-stack.yml"

  lifecycle {
    ignore_changes = all
  }
}

resource "aws_cloudformation_stack" "buildkite_ci_gpu_8c32g_x1" {
  name = "buildkite-ci-gpu-8c32g-x1"

  parameters = merge(local.common_buildkite_stack_params, {
    AgentsPerInstance      = 1
    BuildkiteAgentTags     = "os=linux,machine-type=cpu,instance-size=g4dn.2xlarge"
    BuildkiteQueue         = "gpu-aws-8c32g-x1"
    InstanceType           = "g4dn.2xlarge"
    MaxSize                = 20
    MinSize                = 0
    OnDemandPercentage     = 100
    RootVolumeSize         = 400
    ScaleInIdlePeriod      = 3600
    ScaleOutFactor         = 1.0
    ScaleOutForWaitingJobs = false
    ImageId                = local.buildkite_gpu_ami_id
  })
  capabilities = ["CAPABILITY_AUTO_EXPAND", "CAPABILITY_IAM", "CAPABILITY_NAMED_IAM"]
  template_url = "https://s3.amazonaws.com/buildkite-aws-stack/latest/aws-stack.yml"

  lifecycle {
    ignore_changes = all
  }
}

data "aws_ami" "buildkite_gpu_v1" {
  most_recent = true
  filter {
    name   = "name"
    values = [local.buildkite_ami_gpu_name]
  }
  owners = ["self"]
}


resource "aws_iam_role" "buildkite_access" {
  name = "BuildkiteAccess"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      # Instead of directly attaching the policy to buildkite
      # we have to use assume role such that resources in other
      # accounts can restrict access to certain roles rather than
      # allowing whole of the account.
      # Buildkite autogenerate role names, also at / path and aws
      # conditional string doesn't expose role-name variable
      # it gives role-id only.
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Sid    = "AssumeRole"
        Principal = {
          AWS = [
            "arn:aws:iam::${var.target_account_id}:role/${aws_cloudformation_stack.buildkite_ci_cpu_4c16g_x8.outputs.InstanceRoleName}",
            "arn:aws:iam::${var.target_account_id}:role/${aws_cloudformation_stack.buildkite_ci_cpu_16c64g_x4.outputs.InstanceRoleName}",
            "arn:aws:iam::${var.target_account_id}:role/${aws_cloudformation_stack.buildkite_ci_gpu_16c64g_x4.outputs.InstanceRoleName}",
            "arn:aws:iam::${var.target_account_id}:role/${aws_cloudformation_stack.buildkite_ci_gpu_8c32g_x1.outputs.InstanceRoleName}",
            # Ideally we should be able to limit only buildkite agents to be able to assume this.
            var.target_account_id
          ]
        }
      }
    ]
  })
  managed_policy_arns = [
    aws_iam_policy.buildkite_access.arn, 
  ]
  max_session_duration = 43200
}