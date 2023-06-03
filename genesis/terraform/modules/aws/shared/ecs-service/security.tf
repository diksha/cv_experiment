resource "aws_iam_role" "service" {
  name = "ecs_${var.service_identifier}_role"
  assume_role_policy = data.aws_iam_policy_document.ecs_assume_role_policy.json
  inline_policy {
    name = "secrets-policy"  
    policy = data.aws_iam_policy_document.secrets_policy.json
  }
  
  inline_policy {
    name = "logs-policy"  
    policy = data.aws_iam_policy_document.logs_policy.json
  }
  tags = {
    Environment = var.environment
  }
}

data "aws_iam_policy_document" "ecs_assume_role_policy" {
  statement {
    actions = ["sts:AssumeRole"]
    effect = "Allow"
    sid = ""
    principals {
      type        = "Service"
      identifiers = [ "ecs-tasks.amazonaws.com"]
    }
  }
}

data "aws_iam_policy_document" "secrets_policy" {
  statement {
    actions   = ["secretsmanager:GetSecretValue"]
    effect = "Allow"
    resources = [
          for k, v in aws_secretsmanager_secret.secrets : v.arn
        ]
  }
}
data "aws_iam_policy_document" "logs_policy" {
  statement {
    actions   = ["logs:CreateLogGroup"]
    effect = "Allow"
    resources = ["*"]
  }
}

resource "aws_iam_role_policy_attachment" "ecs_tasks_execution_role" {
  role       =  aws_iam_role.service.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_secretsmanager_secret" "secrets" {
  for_each = { for secret_var in var.secret_vars : secret_var.key => secret_var }
  name     = "ecs-${var.service_identifier}-${each.key}"
  tags = {
    Environment = var.environment
  }
}

resource "aws_secretsmanager_secret_version" "secrets" {
  for_each      = { for secret_var in var.secret_vars : secret_var.key => secret_var }
  secret_id     = aws_secretsmanager_secret.secrets[each.key].id
  secret_string = each.value.value
}
