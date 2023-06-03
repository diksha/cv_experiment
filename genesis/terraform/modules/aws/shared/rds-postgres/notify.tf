module "notify_slack" {
  source  = "terraform-aws-modules/notify-slack/aws"
  version = "~> 4.0"

  create = var.create_rds_alarms_sns_and_slack
  create_sns_topic = true
  sns_topic_name = "${var.db_identifier}-slack-topic"

  slack_webhook_url = var.aws_notifications_slack_app_webhook_url
  slack_channel     = "aws-notifications"
  slack_username    = "sns"
  lambda_function_name = "${var.db_identifier}-slack-sns-topic"
  cloudwatch_log_group_retention_in_days	    = 400
  log_events = true

  recreate_missing_package = false
}
