resource "aws_iam_policy" "developer_access" {
  name        = "DeveloperAccess"
  description = "Developer Access"
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
          "arn:aws:s3:::voxel-storage/*",
          "arn:aws:s3:::voxel-storage",
          "arn:aws:s3:::voxel-perception/*",
          "arn:aws:s3:::voxel-perception",
          "arn:aws:s3:::voxel-infinity-ai-shared/*",
          "arn:aws:s3:::voxel-infinity-ai-shared",
          "arn:aws:s3:::voxel-perception-production-states-events",
          "arn:aws:s3:::voxel-perception-production-states-events/*",
          "arn:aws:s3:::voxel-perception-production-frame-structs",
          "arn:aws:s3:::voxel-perception-production-frame-structs/*",
          "arn:aws:s3:::voxel-production-triton-models",
          "arn:aws:s3:::voxel-production-triton-models/*",
          "arn:aws:s3:::voxel-staging-triton-models",
          "arn:aws:s3:::voxel-staging-triton-models/*",
          "arn:aws:s3:::voxel-experimental-triton-models",
          "arn:aws:s3:::voxel-experimental-triton-models/*",
          # voxeldata bucket is owned by Uber for labeling purposes
          "arn:aws:s3:::voxeldata/*",
          "arn:aws:s3:::voxeldata",
          # for data operations team
          "arn:aws:s3:::${module.voxel_temp_dataops_bucket.bucket_name}",
          "arn:aws:s3:::${module.voxel_temp_dataops_bucket.bucket_name}/*",
          "arn:aws:s3:::voxel-staging-polygon-graph-configs",        
          "arn:aws:s3:::voxel-staging-polygon-graph-configs/*",        
          "arn:aws:s3:::voxel-production-polygon-graph-configs",        
          "arn:aws:s3:::voxel-production-polygon-graph-configs/*",        
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
          "arn:aws:s3:::voxel-portal-dev/*",
          "arn:aws:s3:::voxel-portal-dev",
          "arn:aws:s3:::voxel-metaverse-backup/*",
          "arn:aws:s3:::voxel-metaverse-backup",
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
          "arn:aws:ecr:${var.primary_region}:${var.target_account_id}:repository/voxel-ci/ubuntu",
          "arn:aws:ecr:${var.primary_region}:${var.target_account_id}:repository/lightly/boris-250909/lightly/worker",
          "arn:aws:ecr:${var.primary_region}:${var.target_account_id}:repository/experimental/jorge/try_flink",
          "arn:aws:ecr:${var.primary_region}:${var.target_account_id}:repository/third_party/aws/lambda/noop",
          "arn:aws:ecr:${var.primary_region}:${var.target_account_id}:repository/third_party/aws/lambda/python",
          # Specify repositories instead of wildcard and set resource based policies.
          "arn:aws:ecr:${var.primary_region}:360054435465:repository/*",
          "arn:aws:ecr:${var.primary_region}:115099983231:repository/*",
        ]
      },
      {
        Sid      = "ECRAuthorization"
        Action   = ["ecr:GetAuthorizationToken"],
        Effect   = "Allow",
        Resource = "*"
      },
      {
        Sid = "AllowECRPulls"
        Action = [
          "ecr:BatchGetImage",
          "ecr:GetDownloadUrlForLayer",
        ]
        Effect = "Allow"
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
          "secretsmanager:PutSecretValue",
          "secretsmanager:CreateSecret",
          "secretsmanager:UpdateSecret",
          "secretsmanager:TagResource",
        ],
        Effect = "Allow",
        Resource = [
          "arn:aws:secretsmanager:${var.primary_region}:${var.target_account_id}:secret:*"
        ]
      },
      {
        Sid      = "SecretsManagerList"
        Action   = [
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
        Resource = ["arn:aws:iam::360054435465:role/developer_access_assumable_role"]
      },
      {
        Sid      = "AssumeDevelopmentAccountRole",
        Action   = ["sts:AssumeRole"],
        Effect   = "Allow",
        Resource = ["arn:aws:iam::${var.context.accounts.development.id}:role/developer_access_assumable_role"]
      },
      {
        Sid      = "InvokeDeveloperCASigner",
        Action   = ["lambda:InvokeFunctionUrl"],
        Effect   = "Allow",
        Resource = [aws_lambda_function_url.developer_ca_signer.function_arn]
      }
    ]
    Version = "2012-10-17"
  })
}


resource "aws_iam_role" "temp_for_perception_experimental_devbox" {
  name = "temp_for_perception_experimental_devbox"
  assume_role_policy = jsonencode({
      Version = "2012-10-17"
      Statement = [
        {
          Action = "sts:AssumeRole"
          Effect = "Allow"
          Sid    = ""
          Principal = {
            Service = "ec2.amazonaws.com"
          }
        },
      ]
    })

  managed_policy_arns = [
    aws_iam_policy.developer_access.arn, 
  ]
}

# This will be manually attached from UI as the ec2 instance is not defined in terraform.
resource "aws_iam_instance_profile" "temp_for_perception_experimental_devbox_ec2_profile" {
  name = "temp_for_perception_experimental_devbox_ec2_profile"
  role = aws_iam_role.temp_for_perception_experimental_devbox.name
}

# TODO: Remove this role after bucket moved to Production account
resource "aws_iam_role_policy" "uber_voxeldata_bucket_access_policy" {
  role   = aws_iam_role.uber_voxeldata_bucket_access.id
  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Action   = [
          "s3:PutObject",
          "s3:PutObjectACL"
        ],
        Effect   = "Allow",
        Resource = "arn:aws:s3:::voxeldata/*"
      },
    ]
  })
}

resource "aws_iam_role" "uber_voxeldata_bucket_access" {
  name = "uber_voxeldata_bucket_access"
  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Action = "sts:AssumeRole",
        Effect = "Allow",
        Principal = {
          AWS = "arn:aws:iam::360054435465:role/uber_datapipeline_lambda"
        },
      },
    ]
  })
}
