locals {
  name = "atlantis"
  namespace = local.name
  service_account_name = local.name
}


resource "helm_release" "main" {
  name             = local.name
  namespace        = local.namespace
  chart            = "atlantis"
  repository       = "https://runatlantis.github.io/helm-charts"
  version          = "4.10.3"
  wait             = true
  timeout          = 600
  create_namespace = true
  values = [templatefile("${path.module}/files/service-helm-values.yaml", {
      IRSA_ARN = module.irsa.arn
      SERVICE_ACCOUNT_NAME = local.service_account_name
    })
  ]
  set_sensitive {
    name = "githubApp.key"
    value = var.atlantis.github_app_pem
  }
  set_sensitive {
    name = "githubApp.secret"
    value = var.atlantis.github_app_secret
  }
  set_sensitive {
    name = "serviceAccountSecrets.gcp-sa-json"
    value = var.google_sa_json_terraform_base64_encoded
  }
}

module "irsa" {
  source         = "Young-ook/eks/aws//modules/iam-role-for-serviceaccount"
  version        = "1.7.5"
  namespace      = local.namespace
  serviceaccount = local.service_account_name
  oidc_url       = var.oidc_provider
  oidc_arn       = "arn:aws:iam::${var.account_id}:oidc-provider/${var.oidc_provider}"
  policy_arns = [
    aws_iam_policy.terraform_primary.arn,
  ]
}


data "aws_iam_policy_document" "terraform_primary_policy" {
  statement {
    sid = "AssumeProductionAccountRole"
    actions = ["sts:AssumeRole"]
    resources = [
      "arn:aws:iam::360054435465:role/TerraformAccess"
    ]
  }

  statement {
    sid = "AssumeDevelopementAccountRole"
    actions = ["sts:AssumeRole"]
    resources = [
      "arn:aws:iam::209075043431:role/TerraformAccess"
    ]
  }

  statement {
    sid = "AssumePrimeAccountRole"
    actions = ["sts:AssumeRole"]
    resources = [
      "arn:aws:iam::203670452561:role/TerraformAccess"
    ]
  }

  statement {
    sid = "AssumeStagingAccountRole"
    actions = ["sts:AssumeRole"]
    resources = [
      "arn:aws:iam::115099983231:role/TerraformAccess"
    ]
  }

  statement {
    sid = "AssumeRootAccountRole"
    actions = ["sts:AssumeRole"]
    resources = [
      "arn:aws:iam::667031391229:role/TerraformAccess"
    ]
  }

  statement {
    sid = "AssumePlatformAccountRole"
    actions = ["sts:AssumeRole"]
    resources = [
      "arn:aws:iam::${var.account_id}:role/TerraformAccess"
    ]
  }

  statement {
    sid = "AssumeGalileoAccountRole"
    actions = ["sts:AssumeRole"]
    resources = [
      "arn:aws:iam::610875650709:role/TerraformAccess"
    ]
  }
}


resource "aws_iam_policy" "terraform_primary" {
  name        = "TerraformPrimary"
  description = "Terraform Primary"
  policy = data.aws_iam_policy_document.terraform_primary_policy.json
}

resource "aws_ecr_repository" "atlantis" {
  name                 = "atlantis"
  image_tag_mutability = "IMMUTABLE"
}

resource "aws_ecr_lifecycle_policy" "ecr_policy" {
  repository = aws_ecr_repository.atlantis.name
  policy     = <<EOF
{
    "rules": [
        {
            "rulePriority": 1,
            "description": "Keep last 5 images",
            "selection": {
                "tagStatus": "any",
                "countType": "imageCountMoreThan",
                "countNumber": 5
            },
            "action": {
                "type": "expire"
            }
        }
    ]
}
EOF
}