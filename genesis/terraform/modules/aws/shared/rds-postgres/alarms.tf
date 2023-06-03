# Alarms
# MIT Licensed code taken from  https://github.com/lorenzoaiello/terraform-aws-rds-alarms/blob/master/main.tf

// CPU Utilization

resource "aws_cloudwatch_metric_alarm" "cpu_utilization_too_high" {
  count               = var.create_high_cpu_alarm && var.create_rds_alarms_sns_and_slack ? 1 : 0
  alarm_name          = "${var.db_identifier}-high-cpu-utilization"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = var.evaluation_period
  metric_name         = "CPUUtilization"
  namespace           = "AWS/RDS"
  period              = var.statistic_period
  statistic           = "Average"
  threshold           = var.cpu_utilization_too_high_threshold
  alarm_description   = "Average database CPU utilization is too high."
  alarm_actions       = [module.notify_slack.slack_topic_arn]
  ok_actions          = [module.notify_slack.slack_topic_arn]

  dimensions = {
    DBInstanceIdentifier = module.primary.db_instance_id
  }
}

resource "aws_cloudwatch_metric_alarm" "cpu_credit_balance_too_low" {
  count               = var.create_low_cpu_credit_alarm && var.create_rds_alarms_sns_and_slack ? length(regexall("(t2|t3)", var.db_instance_size)) > 0 ? 1 : 0 : 0
  alarm_name          = "${var.db_identifier}-low-cpu-credit-balance"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = var.evaluation_period
  metric_name         = "CPUCreditBalance"
  namespace           = "AWS/RDS"
  period              = var.statistic_period
  statistic           = "Average"
  threshold           = var.cpu_credit_balance_too_low_threshold
  alarm_description   = "Average database CPU credit balance is too low, a negative performance impact is imminent."
  alarm_actions       = [module.notify_slack.slack_topic_arn]
  ok_actions          = [module.notify_slack.slack_topic_arn]

  dimensions = {
    DBInstanceIdentifier = module.primary.db_instance_id
  }
}

// Disk Utilization
resource "aws_cloudwatch_metric_alarm" "disk_queue_depth_too_high" {
  count               = var.create_high_queue_depth_alarm && var.create_rds_alarms_sns_and_slack ? 1 : 0
  alarm_name          = "${var.db_identifier}-high-disk-queue-depth"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = var.evaluation_period
  metric_name         = "DiskQueueDepth"
  namespace           = "AWS/RDS"
  period              = var.statistic_period
  statistic           = "Average"
  threshold           = var.disk_queue_depth_too_high_threshold
  alarm_description   = "Average database disk queue depth is too high, performance may be negatively impacted."
  alarm_actions       = [module.notify_slack.slack_topic_arn]
  ok_actions          = [module.notify_slack.slack_topic_arn]

  dimensions = {
    DBInstanceIdentifier = module.primary.db_instance_id
  }
}

resource "aws_cloudwatch_metric_alarm" "disk_free_storage_space_too_low" {
  count               = var.create_low_disk_space_alarm && var.create_rds_alarms_sns_and_slack ? 1 : 0
  alarm_name          = "${var.db_identifier}-low-free-storage-space"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = var.evaluation_period
  metric_name         = "FreeStorageSpace"
  namespace           = "AWS/RDS"
  period              = var.statistic_period
  statistic           = "Average"
  threshold           = var.disk_free_storage_space_too_low_threshold
  alarm_description   = "Average database free storage space is too low and may fill up soon."
  alarm_actions       = [module.notify_slack.slack_topic_arn]
  ok_actions          = [module.notify_slack.slack_topic_arn]

  dimensions = {
    DBInstanceIdentifier = module.primary.db_instance_id
  }
}

resource "aws_cloudwatch_metric_alarm" "disk_burst_balance_too_low" {
  count               = var.create_low_disk_burst_alarm && var.create_rds_alarms_sns_and_slack ? 1 : 0
  alarm_name          = "${var.db_identifier}-disk-burst-balance-too-low"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = var.evaluation_period
  metric_name         = "BurstBalance"
  namespace           = "AWS/RDS"
  period              = var.statistic_period
  statistic           = "Average"
  threshold           = var.disk_burst_balance_too_low_threshold
  alarm_description   = "Average database storage burst balance is too low, a negative performance impact is imminent."
  alarm_actions       = [module.notify_slack.slack_topic_arn]
  ok_actions          = [module.notify_slack.slack_topic_arn]

  dimensions = {
    DBInstanceIdentifier = module.primary.db_instance_id
  }
}


// Memory Utilization
resource "aws_cloudwatch_metric_alarm" "memory_freeable_too_low" {
  count               = var.create_low_memory_alarm && var.create_rds_alarms_sns_and_slack ? 1 : 0
  alarm_name          = "${var.db_identifier}-low-freeable-memory"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = var.evaluation_period
  metric_name         = "FreeableMemory"
  namespace           = "AWS/RDS"
  period              = var.statistic_period
  statistic           = "Average"
  threshold           = var.memory_freeable_too_low_threshold
  alarm_description   = "Average database freeable memory is too low, performance may be negatively impacted."
  alarm_actions       = [module.notify_slack.slack_topic_arn]
  ok_actions          = [module.notify_slack.slack_topic_arn]

  dimensions = {
    DBInstanceIdentifier = module.primary.db_instance_id
  }
}

resource "aws_cloudwatch_metric_alarm" "memory_swap_usage_too_high" {
  count               = var.create_swap_alarm && var.create_rds_alarms_sns_and_slack ? 1 : 0
  alarm_name          = "${var.db_identifier}-high-swap-usage"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = var.evaluation_period
  metric_name         = "SwapUsage"
  namespace           = "AWS/RDS"
  period              = var.statistic_period
  statistic           = "Average"
  threshold           = var.memory_swap_usage_too_high_threshold
  alarm_description   = "Average database swap usage is too high, performance may be negatively impacted."
  alarm_actions       = [module.notify_slack.slack_topic_arn]
  ok_actions          = [module.notify_slack.slack_topic_arn]

  dimensions = {
    DBInstanceIdentifier = module.primary.db_instance_id
  }
}

// Connection Count
resource "aws_cloudwatch_metric_alarm" "connection_count_anomalous" {
  count               = var.create_anomaly_alarm && var.create_rds_alarms_sns_and_slack ? 1 : 0
  alarm_name          = "${var.db_identifier}-high-anamalous-connection-count"
  comparison_operator = "GreaterThanUpperThreshold"
  evaluation_periods  = var.evaluation_period
  threshold_metric_id = "e1"
  alarm_description   = "Anomalous database connection count detected. Something unusual is happening."
  alarm_actions       = [module.notify_slack.slack_topic_arn]
  ok_actions          = [module.notify_slack.slack_topic_arn]

  metric_query {
    id          = "e1"
    expression  = "ANOMALY_DETECTION_BAND(m1, ${var.anomaly_band_width})"
    label       = "DatabaseConnections (Expected)"
    return_data = "true"
  }

  metric_query {
    id          = "m1"
    return_data = "true"
    metric {
      metric_name = "DatabaseConnections"
      namespace   = "AWS/RDS"
      period      = var.anomaly_period
      stat        = "Average"
      unit        = "Count"

      dimensions = {
        DBInstanceIdentifier = module.primary.db_instance_id
      }
    }
  }
}

resource "aws_cloudwatch_metric_alarm" "read_iops_too_high" {
  count               = var.create_read_iops_alarm && var.create_rds_alarms_sns_and_slack ? 1 : 0
  alarm_name          = "${var.db_identifier}-high-read-iops"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = var.evaluation_period
  metric_name         = "ReadIOPS"
  namespace           = "AWS/RDS"
  period              = var.statistic_period
  statistic           = "Average"
  threshold           = var.read_iops_too_high_threshold
  alarm_description   = "Average database read iops is too high, performance may be negatively impacted."
  alarm_actions       = [module.notify_slack.slack_topic_arn]
  ok_actions          = [module.notify_slack.slack_topic_arn]

  dimensions = {
    DBInstanceIdentifier = module.primary.db_instance_id
  }
}

resource "aws_cloudwatch_metric_alarm" "maximum_used_transaction_ids_too_high" {
  count               = contains(["aurora-postgresql", "postgres"], module.primary.db_instance_engine) && var.create_rds_alarms_sns_and_slack ? 1 : 0
  alarm_name          = "${var.db_identifier}-high-maximum-used-transaction-ids"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = var.evaluation_period
  metric_name         = "MaximumUsedTransactionIDs"
  namespace           = "AWS/RDS"
  period              = var.statistic_period
  statistic           = "Average"
  threshold           = var.maximum_used_transaction_ids_too_high_threshold
  alarm_description   = "Nearing a possible critical transaction ID wraparound."
  alarm_actions       = [module.notify_slack.slack_topic_arn]
  ok_actions          = [module.notify_slack.slack_topic_arn]

  dimensions = {
    DBInstanceIdentifier = module.primary.db_instance_id
  }
}
