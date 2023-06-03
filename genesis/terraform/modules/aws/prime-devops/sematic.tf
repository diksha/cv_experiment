module "s3_bucket_for_sematic" {
  source            = "../shared/s3-bucket"
  target_account_id = var.target_account_id
  bucket_name       = "${var.target_account_id}-sematic-ci-cd"
  primary_region    = var.primary_region
}

resource "aws_iam_policy" "sematic_s3_bucket_policy" {
  name        = "${module.s3_bucket_for_sematic.bucket_id}-access"
  path        = "/"
  description = "${module.s3_bucket_for_sematic.bucket_id} Access"
  policy      = data.aws_iam_policy_document.sematic_s3_policy_document.json
}

data "aws_iam_policy_document" "sematic_s3_policy_document" {
  statement {
    actions = [
      "s3:*",
    ]
    resources = [
      module.s3_bucket_for_sematic.bucket_arn,
      "${module.s3_bucket_for_sematic.bucket_arn}/*"
    ]
    effect = "Allow"
  }
}

module "sematic_service" {
  providers = {
    aws = aws
    grafana = grafana
  }
  source                                  = "./modules/sematic"
  account_id                              = var.target_account_id
  eks_cluster_name                        = module.base-account-setup.eks_cluster_name
  oidc_provider                           = local.oidc_provider
  primary_region                          = var.primary_region
  perception_verbose_slack_hook_url_token = var.perception_verbose_slack_hook_url_token
  google_client_id                        = var.google_client_id
  environment                             = var.environment
  private_subnet_ids                      = module.vpc.private_subnet_ids
  private_subnet_cidrs                    = module.vpc.private_subnet_cidrs
  vpc_id                                  = module.vpc.vpc_id
}

resource "aws_iam_policy" "sematic_access" {
  name        = "SematicAccess"
  description = "Access for Engineers from Sematic and the Workers"
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
          "s3:PutObject",
          "s3:RestoreObject",
          "s3:AbortMultipartUpload",
          "s3:DeleteObject",
          "s3:GetBucketVersioning",
          "s3:GetLifecycleConfiguration",
          "s3:GetEncryptionConfiguration",
          "s3:GetBucketTagging",
          "s3:GetBucketPolicy",
          "s3:GetBucketAcl",
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
          "arn:aws:s3:::voxel-perception/*",
          "arn:aws:s3:::voxel-perception",
          # voxeldata bucket is owned by Uber for labeling purposes
          "arn:aws:s3:::voxeldata/*",
          "arn:aws:s3:::voxeldata",
          # for data operations team
          "arn:aws:s3:::${module.voxel_temp_dataops_bucket.bucket_name}",
          "arn:aws:s3:::${module.voxel_temp_dataops_bucket.bucket_name}/*"          
        ]
      },
      {
        Sid = "ListBuckets"
        Action = [
          "s3:GetBucketLocation",
          "s3:ListAllMyBuckets",
        ],
        Effect = "Allow",
        Resource = [
          "*"
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
          "arn:aws:s3:::voxel-portal-production-mumbai/*",
          "arn:aws:s3:::voxel-portal-production-mumbai",
          "arn:aws:s3:::voxel-portal-production/*",
          "arn:aws:s3:::voxel-portal-production",
          "arn:aws:s3:::voxel-portal-staging/*",
          "arn:aws:s3:::voxel-portal-staging",
          "arn:aws:s3:::voxel-storage/*",
          "arn:aws:s3:::voxel-storage",
        ]
      },
      {
        Sid = "EKS"
        Action = [
          "eks:ListFargateProfiles",
          "eks:DescribeNodegroup",
          "eks:ListNodegroups",
          "eks:ListUpdates",
          "eks:AccessKubernetesApi",
          "eks:ListAddons",
          "eks:DescribeCluster",
          "eks:DescribeAddonVersions",
          "eks:ListClusters",
          "eks:ListIdentityProviderConfigs",
        ]
        Effect = "Allow",
        Resource = [
          "arn:aws:eks:${var.primary_region}:${var.target_account_id}:nodegroup/jenkins/*/*",
          "arn:aws:eks:${var.primary_region}:${var.target_account_id}:cluster/jenkins"
        ]
      },
      {
        Sid      = "EKSList"
        Action   = ["eks:ListClusters"],
        Effect   = "Allow",
        Resource = "*"
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
          "arn:aws:ecr:${var.primary_region}:${var.target_account_id}:repository/lightly/boris-250909/lightly/worker",
          "arn:aws:ecr:${var.primary_region}:${var.target_account_id}:repository/voxel-ci/ubuntu"
        ]
      },
      {
        Sid      = "ECRAuthorization"
        Action   = ["ecr:GetAuthorizationToken"],
        Effect   = "Allow",
        Resource = "*"
      },
      {
        Sid = "ECRDescribeRepositories"
        Action = [
          "ecr:DescribeRepositories",
        ],
        Effect = "Allow",
        Resource = [
          "arn:aws:ecr:${var.primary_region}:${var.target_account_id}:repository/*"
        ]
      },
      {
        Sid = "SecretsManager"
        Action = [
          "secretsmanager:GetSecretValue",
        ],
        Effect = "Allow",
        Resource = [
          "arn:aws:secretsmanager:${var.primary_region}:${var.target_account_id}:secret:*"
        ]
      },
      {
        Sid = "SecretsManagerUpdate"
        Action = [
          "secretsmanager:UpdateSecret",
        ],
        Effect = "Allow",
        Resource = [
          "arn:aws:secretsmanager:${var.primary_region}:${var.target_account_id}:secret:PERCEPTION_PORTAL_AUTH_TOKEN-JETkNa"
        ]
      },
      {
        Sid = "SecretsManagerList"
        Action = [
          "secretsmanager:ListSecrets",
          "secretsmanager:DescribeSecret",
        ],
        Effect   = "Allow",
        Resource = "*"
      },
      {
        Sid      = "AssumeProductionAccountRole"
        Action   = ["sts:AssumeRole"],
        Effect   = "Allow",
        Resource = ["arn:aws:iam::360054435465:role/sematic_access_assumable_role"]
      },
    ]
    Version = "2012-10-17"
  })
}
