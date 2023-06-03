locals {
  standard_images_policy = jsonencode(
    {
      Version = "2012-10-17",
      Statement = {
        Sid = "primary",
        Action = [
          "ecr:GetDownloadUrlForLayer", "ecr:BatchGetImage",
          "ecr:BatchCheckLayerAvailability", "ecr:PutImage",
          "ecr:InitiateLayerUpload", "ecr:UploadLayerPart",
          "ecr:CompleteLayerUpload", "ecr:DescribeRepositories",
          "ecr:GetRepositoryPolicy", "ecr:ListImages",
          "ecr:DeleteRepository", "ecr:BatchDeleteImage",
          "ecr:SetRepositoryPolicy", "ecr:DeleteRepositoryPolicy"
        ]
        Effect = "Allow"
        # Change from root to specific users/roles.
        Principal = {
          AWS = [
            "arn:aws:iam::${var.prime_account_id}:root",
            "arn:aws:iam::${var.prime_account_id}:role/BuildkiteAccess",
            "arn:aws:iam::${var.platform_account_id}:root",
            "arn:aws:iam::${var.production_account_id}:root",
            "arn:aws:iam::${var.root_account_id}:role/buildkite_access_assumable_role"
          ]
        }
      }
    }
  )
}

# Move to prime module.
resource "aws_ecr_repository" "voxel_ci_ubuntu" {
  provider             = aws.prime
  name                 = "voxel-ci/ubuntu"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_repository_policy" "voxel_ci_ubuntu" {
  provider   = aws.prime
  repository = aws_ecr_repository.voxel_ci_ubuntu.name
  policy     = local.standard_images_policy
}

resource "aws_ecr_repository" "experimental_jorge_try_flink" {
  provider             = aws.prime
  name                 = "experimental/jorge/try_flink"
  image_tag_mutability = "IMMUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_repository_policy" "experimental_jorge_try_flink" {
  provider   = aws.prime
  repository = aws_ecr_repository.experimental_jorge_try_flink.name
  policy     = local.standard_images_policy
}
resource "aws_ecr_repository" "lambda_python_baseimage" {
    provider = aws.prime
    name = "third_party/aws/lambda/python"
    image_tag_mutability = "IMMUTABLE"

    image_scanning_configuration {
        scan_on_push = true
    }
}

resource "aws_ecr_repository_policy" "lambda_python_baseimage" {
    provider   = aws.prime
    repository = aws_ecr_repository.lambda_python_baseimage.name
    policy     = local.standard_images_policy
}