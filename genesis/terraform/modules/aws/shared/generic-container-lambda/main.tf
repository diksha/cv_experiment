locals {
    noop_image_uri = "203670452561.dkr.ecr.us-west-2.amazonaws.com/third_party/aws/lambda/noop:latest"
}

resource "aws_iam_role" "main" {
  name = var.iam_role_name_override != "" ? var.iam_role_name_override : "${var.function_name}-lambda-execution-role"
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
  managed_policy_arns = concat([
    var.vpc.enabled ? "arn:aws:iam::aws:policy/service-role/AWSLambdaVPCAccessExecutionRole" : "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
  ], var.extra_policy_arns)
}

resource "aws_ecr_repository" "main" {
  name = var.repository_name
  image_tag_mutability = "IMMUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_lambda_function" "main" {
  function_name = var.function_name
  role = aws_iam_role.main.arn
  package_type = "Image"
  image_uri = local.noop_image_uri
  timeout = var.timeout

  memory_size = var.memory_size

  environment {
    variables = {
      ENVIRONMENT = var.context.environment
    }
  }

  vpc_config {
    subnet_ids = var.vpc.enabled ? var.vpc.subnet_ids : []
    security_group_ids = var.vpc.enabled ? var.vpc.security_group_ids : []
  }

  lifecycle {
    ignore_changes = [
      image_uri
    ]
  }
}
