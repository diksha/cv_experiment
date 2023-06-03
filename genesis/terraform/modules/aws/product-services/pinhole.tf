# Pinhole is a service for the edge to be able to reach our AWS
# environment via HTTPS proxy to enable edges in more restrictive environments

locals {
  pinhole_dns_names = {
    "production": "edge.voxelai.com"
    "staging": "pinhole.staging.voxelplatform.com"
  }
}

resource "random_password" "pinholecert_ca_password" {
    length = 16
    special = false
}

module "pinholecert_ca" {
    source = "../shared/eks-autocert-mtls/intermediate-ca"
    root_ca = var.services_root_ca
    password = random_password.pinholecert_ca_password.result
    common_name = "Voxel Edge Pinhole CA"
}

data "archive_file" "pinholecert_init" {
  type = "zip"
  output_path = "${path.module}/lambda_function_payload.zip"

  source {
    content = "hello"
    filename = "dummy.txt"
  }
}

resource "aws_secretsmanager_secret" "pinholecert" {
    name = "lambdas/pinholecert"
}

resource "aws_secretsmanager_secret_version" "pinholecert" {
    secret_id = aws_secretsmanager_secret.pinholecert.id
    secret_string = jsonencode({
        "root_ca.crt" = var.services_root_ca.cert_pem
        "intermediate_ca.crt" = module.pinholecert_ca.cert_pem
        "intermediate_ca.key" = module.pinholecert_ca.key_pem
        "password" = random_password.pinholecert_ca_password.result
    })
}

data "aws_iam_policy_document" "pinholecert_access" {
  statement {
    sid = "GetSecretValue"
    actions = ["secretsmanager:GetSecretValue"]
    resources = [aws_secretsmanager_secret.pinholecert.arn]
  }

  statement {
    sid = "GetThingFromCertificate"
    actions = [
      "iot:DescribeCertificate",
      "iot:ListPrincipalThings",
    ]
    resources = ["*"]
  }
}

resource "aws_iam_policy" "pinholecert_access" {
    name = "PinholecertAccess"
    policy = data.aws_iam_policy_document.pinholecert_access.json
}

resource "aws_iam_role" "pinholecert" {
  name = "Pinholecert"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Sid    = ""
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      },
    ]
  })
  managed_policy_arns = [
    "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
    aws_iam_policy.pinholecert_access.arn,
  ]
}


resource "aws_lambda_function" "pinholecert" {
  function_name = "pinholecert-${local.environment}"
  filename = data.archive_file.pinholecert_init.output_path
  role = aws_iam_role.pinholecert.arn
  runtime = "go1.x"
  handler = "pinholecert"
  timeout = 30

  environment {
    variables = {
      PINHOLECERT_SECRET_ARN = aws_secretsmanager_secret.pinholecert.arn
    }
  }
}

resource "aws_lambda_function_url" "pinholecert" {
    function_name = aws_lambda_function.pinholecert.function_name
    authorization_type = "AWS_IAM"
}

module "pinhole" {
  source = "./modules/generic-service"
  name = "pinhole"
  context = var.context
  initial_image_tag = "4cf082ff958689cc90b7b9ccb5a466aab3f9e91cf91fcbade2fe3acbaebd7c52"
  oidc_provider = local.oidc_provider
  tls_sans = [local.pinhole_dns_names[local.environment]]
}

resource "kubernetes_service_v1" "pinhole_external" {
  metadata {
    name = "pinhole-external"
    namespace = module.pinhole.namespace
    annotations = {
      "service.beta.kubernetes.io/aws-load-balancer-scheme" = "internet-facing"
      "service.beta.kubernetes.io/aws-load-balancer-type" = "external"
      "service.beta.kubernetes.io/aws-load-balancer-nlb-target-type" = "ip"
      "service.beta.kubernetes.io/aws-load-balancer-additional-resource-tags" = "map-migrated=d-server-00swbp99drezfh"
    }
  }
  spec {
    selector = {
      "app.kubernetes.io/name" = "pinhole"
    }
    port {
      port        = 443
      target_port = 8443
    }
    type = "LoadBalancer"
  }
}