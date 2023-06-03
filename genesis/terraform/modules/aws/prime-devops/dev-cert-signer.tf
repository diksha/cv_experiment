resource "random_password" "developer_ca_password" {
    length = 16
    special = false
}

module "developer_ca" {
    source = "../shared/eks-autocert-mtls/intermediate-ca"
    root_ca = var.services_root_ca
    password = random_password.developer_ca_password.result
    common_name = "Voxel Services Developer Access CA"
}

data "archive_file" "developer_ca_signer_init" {
  type = "zip"
  output_path = "${path.module}/lambda_function_payload.zip"

  source {
    content = "hello"
    filename = "dummy.txt"
  }
}

resource "aws_secretsmanager_secret" "developer_ca_signer" {
    name = "lambdas/developer-ca-signer"
}

resource "aws_secretsmanager_secret_version" "developer_ca_signer" {
    secret_id = aws_secretsmanager_secret.developer_ca_signer.id
    secret_string = jsonencode({
        "root_ca.crt" = var.services_root_ca.cert_pem
        "intermediate_ca.crt" = module.developer_ca.cert_pem
        "intermediate_ca.key" = module.developer_ca.key_pem
        "password" = random_password.developer_ca_password.result
    })
}

data "aws_iam_policy_document" "developer_ca_signer_access" {
    statement {
        sid = "GetSecretValue"
        actions = ["secretsmanager:GetSecretValue"]
        resources = [aws_secretsmanager_secret.developer_ca_signer.arn]
    }
}

resource "aws_iam_policy" "developer_ca_signer_access" {
    name = "DeveloperCASignerAccess"
    policy = data.aws_iam_policy_document.developer_ca_signer_access.json
}

resource "aws_iam_role" "developer_ca_signer" {
  name = "DeveloperCASigner"
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
    aws_iam_policy.developer_ca_signer_access.arn,
  ]
}


resource "aws_lambda_function" "developer_ca_signer" {
  function_name = "developer-ca-signer"
  filename = data.archive_file.developer_ca_signer_init.output_path
  role = aws_iam_role.developer_ca_signer.arn
  runtime = "go1.x"
  handler = "developer-ca-signer"
  timeout = 30

  environment {
    variables = {
      DEVCERT_SECRET_ARN = aws_secretsmanager_secret.developer_ca_signer.arn
    }
  }
}

resource "aws_lambda_function_url" "developer_ca_signer" {
    function_name = aws_lambda_function.developer_ca_signer.function_name
    authorization_type = "AWS_IAM"
}