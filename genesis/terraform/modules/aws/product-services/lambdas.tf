moved {
  from = aws_iam_role.states_events_rollup
  to = module.states_events_rollup.aws_iam_role.main
}

moved {
  from = aws_ecr_repository.states_events_rollup
  to = module.states_events_rollup.aws_ecr_repository.main
}

moved {
  from = aws_lambda_function.states_events_rollup
  to = module.states_events_rollup.aws_lambda_function.main
}

locals {
    is_production = var.context.environment == "production" ? 1 : 0

}
module "states_events_rollup" {
  source = "../shared/generic-container-lambda"
  context = var.context

  memory_size = 512
  timeout = 300
  extra_policy_arns = [aws_iam_policy.states_events_rollup.arn]

  function_name = "portal-${local.environment}-states-events-rollup"
  repository_name = "services/portal/lambdas/states_events_rollup"
  iam_role_name_override = "PortalStatesEventsRollup-${local.environment}"
  vpc = {
    enabled = true
    security_group_ids = [aws_security_group.states_events_rollup.id]
    subnet_ids = var.context.vpc.private_subnet_ids
  }

}
# read only pdb (portal_secret_read) vs. read and write from pdb (states_and_events_rollup)
data "aws_iam_policy_document" "states_events_rollup" {
  statement {
    sid = "StatesEventsBucketRead"
    actions = [
      "s3:AbortMultipartUpload",
      "s3:GetBucketLocation",
      "s3:GetObject",
      "s3:ListBucket",
      "s3:ListBucketMultipartUploads",
      "s3:PutObject"
    ]
    resources = [
      "arn:aws:s3:::voxel-perception-${local.environment}-states-events",
      "arn:aws:s3:::voxel-perception-${local.environment}-states-events/*",
    ]
  }

  statement {
    sid = "PortalSecretRead"
    actions = [
      "secretsmanager:GetSecretValue",
    ]
    resources = [
      var.portal_secret_arn
    ]
  }
}

resource "aws_iam_policy" "states_events_rollup" {
  name = "PortalStatesEventsRollup-${local.environment}"
  description = "allows the portal states events rollup lambda function access to aws resources"
  policy = data.aws_iam_policy_document.states_events_rollup.json
}

resource "aws_security_group" "states_events_rollup" {
  name = "portal-${local.environment}-states-events-rollup"
  description = "Allows access to resources in our vpc"
  vpc_id = var.context.vpc.id
  egress {
    from_port        = 0
    to_port          = 0
    protocol         = "-1"
    cidr_blocks      = ["0.0.0.0/0"]
    ipv6_cidr_blocks = ["::/0"]
  }
}

resource "aws_s3_bucket_notification" "states_events_rollup_trigger" {
  bucket = "voxel-perception-${local.environment}-states-events"
  lambda_function {
    lambda_function_arn = module.states_events_rollup.function_arn
    events = ["s3:ObjectCreated:*"]
  }
}

resource "aws_lambda_permission" "states_events_rollup_permission" {
  statement_id = "AllowS3Invoke"
  action = "lambda:InvokeFunction"
  function_name = module.states_events_rollup.function_name
  principal = "s3.amazonaws.com"
  source_arn = "arn:aws:s3:::voxel-perception-${local.environment}-states-events"
}

module "vper_precomputer" {
  source = "../shared/generic-container-lambda"
  context = var.context

  memory_size = 512
  timeout = 300
  extra_policy_arns = [aws_iam_policy.portal_secret_read.arn]

  function_name = "VperPrecomputer-${local.environment}"
  repository_name = "services/portal/lambdas/vper_precomputer"
  iam_role_name_override = "VperPrecomputer-${local.environment}"
  vpc = {
    enabled = true
    # TODO(itay): Modularize the aws_security_group resource
    security_group_ids = [aws_security_group.states_events_rollup.id]
    subnet_ids = var.context.vpc.private_subnet_ids
  }
}

data "aws_iam_policy_document" "portal_secret_read" {
  statement {
    sid = "PortalSecretRead"
    actions = [
      "secretsmanager:GetSecretValue",
    ]
    resources = [
      var.portal_secret_arn
    ]
  }
}

resource "aws_iam_policy" "portal_secret_read" {
  name = "PortalSecretRead-${local.environment}"
  description = "Allows reading of Portal's Secrets Manager"
  policy = data.aws_iam_policy_document.portal_secret_read.json
}

module "user_session" {
  source = "../shared/generic-container-lambda"
  context = var.context

  memory_size = 512
  timeout = 300
  # Using the states_events_rollup policy for now, until we have a dedicated policy for this lambda if needed.
  extra_policy_arns = [aws_iam_policy.states_events_rollup.arn]

  function_name = "portal-${local.environment}-user-session"
  repository_name = "services/portal/lambdas/user_session"
  iam_role_name_override = "PortalUserSession-${local.environment}"
  vpc = {
    enabled = true
    security_group_ids = [aws_security_group.states_events_rollup.id]
    subnet_ids = var.context.vpc.private_subnet_ids
  }  
}

resource "aws_iam_role_policy" "uber_datapipeline_lambda_policy" {
  count  = local.is_production

  role = aws_iam_role.uber_datapipeline_lambda[0].id
  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Action   = "sts:AssumeRole",
        Effect   = "Allow",
        Resource = "arn:aws:iam::203670452561:role/uber_voxeldata_bucket_access"
      },
      {
        Action = "secretsmanager:GetSecretValue",
        Effect = "Allow",
        Resource = var.portal_secret_arn
      }
    ]
  })
}

resource "aws_iam_role" "uber_datapipeline_lambda" {
  count  = local.is_production

  name = "uber_datapipeline_lambda"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole",
        Effect = "Allow",
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      },
    ]
  })
  managed_policy_arns = [
    "arn:aws:iam::aws:policy/service-role/AWSLambdaVPCAccessExecutionRole"
  ]
}

resource "aws_lambda_function" "uber_data_pipeline" {
  count  = local.is_production

  function_name = "uber-data-pipeline"
  
  role = aws_iam_role.uber_datapipeline_lambda[0].arn
  package_type = "Image"
  image_uri = "360054435465.dkr.ecr.us-west-2.amazonaws.com/production/portal/api:e8e4bbe2a234336f93a5ba48496a5f360788d434"
  timeout = 300

  memory_size = 512

  image_config {
    command = ["/app/core/portal/tools/uber_data_cronjob"]
  }


  environment {
    variables = {
      ENVIRONMENT = var.context.environment
    }
  }

  vpc_config {
    subnet_ids = var.context.vpc.private_subnet_ids
    security_group_ids = [aws_security_group.states_events_rollup.id]
  }

  lifecycle {
    ignore_changes = [
      image_uri
    ]
  }
}

resource "aws_cloudwatch_event_rule" "schedule_hourly" {
  count  = local.is_production

  name        = "RunUberDataPipelineHourly"
  description = "Trigger the Uber Data Pipeline Lambda function hourly"

  schedule_expression = "rate(1 hour)"
}

resource "aws_lambda_permission" "allow_cloudwatch" {
  count  = local.is_production

  statement_id  = "AllowExecutionFromCloudWatch"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.uber_data_pipeline[0].function_name
  principal     = "events.amazonaws.com"
}

resource "aws_cloudwatch_event_target" "schedule_hourly_target" {
  count  = local.is_production

  rule      = aws_cloudwatch_event_rule.schedule_hourly[0].name
  target_id = "RunUberDataPipelineTarget"
  arn       = aws_lambda_function.uber_data_pipeline[0].arn
}
