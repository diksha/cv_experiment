resource "aws_cloudwatch_event_rule" "main" {
  name          = "guardduty-events"
  description   = "AWS GuardDuty events"
  event_pattern = <<EOF
{
  "detail-type": [
    "GuardDuty Finding"
  ],
  "source": [
    "aws.guardduty"
  ]
}
EOF
}

resource "aws_cloudwatch_event_rule" "main_east_1" {
  provider      = aws.us-east-1
  name          = "guardduty-events"
  description   = "AWS GuardDuty events"
  event_pattern = <<EOF
{
  "detail-type": ["GuardDuty Finding"],
  "source": ["aws.guardduty"]
}
EOF
}


resource "aws_cloudwatch_event_target" "slack" {
  rule      = aws_cloudwatch_event_rule.main.name
  target_id = "send-to-sns-slack"
  arn       = module.notify_slack.slack_topic_arn
  input_transformer {
    input_paths = {
      title       = "$.detail.title"
      description = "$.detail.description"
      eventTime   = "$.detail.service.eventFirstSeen"
      region      = "$.detail.region"
    }
    input_template = "\"GuardDuty events in <region> first seen at <eventTime>: <title> <description>\""
  }
}


resource "aws_cloudwatch_event_target" "slack_east_1" {
  provider  = aws.us-east-1
  rule      = aws_cloudwatch_event_rule.main.name
  target_id = "send-to-sns-slack"
  arn       = module.notify_slack.slack_topic_arn
  input_transformer {
    input_paths = {
      title       = "$.detail.title"
      description = "$.detail.description"
      eventTime   = "$.detail.service.eventFirstSeen"
      region      = "$.detail.region"
    }
    input_template = "\"GuardDuty events in <region> first seen at <eventTime>: <title> <description>\""
  }
}


module "notify_slack" {
  source                                 = "terraform-aws-modules/notify-slack/aws"
  version                                = "5.6.0"
  create                                 = true
  create_sns_topic                       = true
  sns_topic_name                         = "guard-duty-slack-topic"
  slack_webhook_url                      = var.aws_notifications_slack_app_webhook_url
  slack_channel                          = "aws-notifications"
  slack_username                         = "sns"
  lambda_function_name                   = "guard-duty-slack-sns-topic"
  cloudwatch_log_group_retention_in_days = 400
  log_events                             = true
}


module "notify_slack_east_1" {
  providers = {
    aws = aws.us-east-1
  }
  source                                 = "terraform-aws-modules/notify-slack/aws"
  version                                = "5.6.0"
  create                                 = true
  create_sns_topic                       = true
  sns_topic_name                         = "guard-duty-slack-topic"
  slack_webhook_url                      = var.aws_notifications_slack_app_webhook_url
  slack_channel                          = "aws-notifications"
  slack_username                         = "sns"
  lambda_function_name                   = "guard-duty-east-1-slack-sns-topic"
  cloudwatch_log_group_retention_in_days = 400
  log_events                             = true
}

resource "aws_cloudwatch_metric_alarm" "failed_invocation" {
  alarm_name          = "failed-invocation-guard-duty-findings"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 60
  metric_name         = "FailedInvocations"
  namespace           = "AWS/Events"
  period              = 300
  statistic           = "Sum"
  threshold           = 0
  alarm_description   = "Failed invocations for GuardDuty Events"
  alarm_actions       = [module.notify_slack.slack_topic_arn]
  ok_actions          = [module.notify_slack.slack_topic_arn]

  dimensions = {
    RuleName = "guardduty-events"
  }
}

resource "aws_cloudwatch_metric_alarm" "failed_invocation_east_1" {
  provider            = aws.us-east-1
  alarm_name          = "failed-invocation-guard-duty-findings"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 60
  metric_name         = "FailedInvocations"
  namespace           = "AWS/Events"
  period              = 300
  statistic           = "Sum"
  threshold           = 0
  alarm_description   = "Failed invocations for GuardDuty Events"
  alarm_actions       = [module.notify_slack_east_1.slack_topic_arn]
  ok_actions          = [module.notify_slack_east_1.slack_topic_arn]

  dimensions = {
    RuleName = "guardduty-events"
  }
}
