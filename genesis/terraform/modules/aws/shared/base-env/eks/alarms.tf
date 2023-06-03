module "notify_slack" {
  source                                 = "terraform-aws-modules/notify-slack/aws"
  version                                = "~> 4.0"
  create                                 = var.create_alarm
  create_sns_topic                       = true
  sns_topic_name                         = "${var.vpc_name}-${var.cluster_name}-slack-topic"
  slack_webhook_url                      = var.aws_notifications_slack_app_webhook_url
  slack_channel                          = "aws-notifications"
  slack_username                         = "sns"
  lambda_function_name                   = "${var.vpc_name}-${var.cluster_name}-slack-sns-topic"
  cloudwatch_log_group_retention_in_days = 400
  log_events                             = true
  recreate_missing_package               = false
}


resource "aws_cloudwatch_metric_alarm" "autoscaling_group_high_cpu_usage" {
  for_each            = var.create_alarm ? { for idx, value in flatten([var.should_create_standard_node_group ? ["standard_node_group"] : [], var.should_create_gpu_node_group ? ["gpu_node_group"] : [], keys(var.extra_node_groups)]) : idx => module.eks.eks_managed_node_groups_autoscaling_group_names[idx] } : {}
  alarm_name          = "${each.value}-high-cpu-usage"
  alarm_description   = "This metric monitors ec2 cpu utilization"
  comparison_operator = "GreaterThanOrEqualToThreshold"
  evaluation_periods  = "2"
  metric_name         = "CPUUtilization"
  namespace           = "AWS/EC2"
  period              = "120"
  statistic           = "Average"
  threshold           = "90"
  alarm_actions       = [module.notify_slack.slack_topic_arn]
  ok_actions          = [module.notify_slack.slack_topic_arn]
  dimensions = {
    AutoScalingGroupName = each.value
  }
}
