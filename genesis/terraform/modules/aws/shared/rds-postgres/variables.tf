variable "db_identifier" {
  type = string
}
variable "db_instance_size" {
  description = "the type of instance for Postgres cluster "
  type        = string
  default     = "db.t3.small"
}
variable "dev_mode" {
  default = false
}
# variable "security_group_ids" {}
variable "subnet_ids" {}
variable "environment" {}

variable "existing_db_subnet" {
  default = null
}

variable "vpc_id" {
  type = string
}
variable "ingress_cidr_blocks" {
  default = []
}

variable "replica_vpc_cidr_root" {
  type    = string
  default = ""
}

variable "replica_region" {
  type    = string
  default = ""
}

variable "replica_vpc_name" {
  type    = string
  default = ""
}

variable "account_id" {
  type = string
}

variable "live_replica" {
  default = false
}

variable "is_publically_accessible" {
  default = false
}

variable "allocated_storage" {
  default = 20
}

variable "performance_insights_enabled" {
  default = true
}

variable "performance_insights_retention_period" {
  default = 7
}

variable "monitoring_interval" {
  default = 60
}

variable "multi_az" {
  default = true
}

variable "backup_replica" {
  default = true
}

variable "evaluation_period" {
  type        = string
  default     = "5"
  description = "The evaluation period over which to use when triggering alarms."
}

variable "statistic_period" {
  type        = string
  default     = "60"
  description = "The number of seconds that make each statistic period."
}

variable "create_high_cpu_alarm" {
  type        = bool
  default     = true
  description = "Whether or not to create the high cpu alarm.  Default is to create it"
}

variable "create_low_cpu_credit_alarm" {
  type        = bool
  default     = true
  description = "Whether or not to create the low cpu credit alarm.  Default is to create it"
}

variable "create_high_queue_depth_alarm" {
  type        = bool
  default     = true
  description = "Whether or not to create the high queue depth alarm.  Default is to create it"
}

variable "create_low_disk_space_alarm" {
  type        = bool
  default     = true
  description = "Whether or not to create the low disk space alarm.  Default is to create it"
}

variable "create_low_disk_burst_alarm" {
  type        = bool
  default     = true
  description = "Whether or not to create the low disk burst alarm.  Default is to create it"
}

variable "create_low_memory_alarm" {
  type        = bool
  default     = true
  description = "Whether or not to create the low memory free alarm.  Default is to create it"
}

variable "create_swap_alarm" {
  type        = bool
  default     = true
  description = "Whether or not to create the high swap usage alarm.  Default is to create it"
}

variable "create_anomaly_alarm" {
  type        = bool
  default     = false
  description = "Whether or not to create the fairly noisy anomaly alarm.  Default is to create it, but recommended to disable this for non-production databases"
}

variable "create_read_iops_alarm" {
  type        = bool
  default     = true
  description = "Whether or not to create read iops alarm. Default to create it."
}

variable "anomaly_period" {
  type        = string
  default     = "600"
  description = "The number of seconds that make each evaluation period for anomaly detection."
}

variable "anomaly_band_width" {
  type        = string
  default     = "2"
  description = "The width of the anomaly band, default 2.  Higher numbers means less sensitive."
}

variable "actions_alarm" {
  type        = list
  default     = []
  description = "A list of actions to take when alarms are triggered. Will likely be an SNS topic for event distribution."
}

variable "actions_ok" {
  type        = list
  default     = []
  description = "A list of actions to take when alarms are cleared. Will likely be an SNS topic for event distribution."
}

variable "cpu_utilization_too_high_threshold" {
  default     = 80
  description = "Alarm threshold for the high-cpu-utilization alarm"
}

variable "cpu_credit_balance_too_low_threshold" {
  default     = 20
  description = "Alarm threshold for the low-cpu-credit-balance alarm"
}

variable "disk_queue_depth_too_high_threshold" {
  default     = 64
  description = "Alarm threshold for the high-disk-queue-depth alarm"
}

variable "disk_free_storage_space_too_low_threshold" {
  default     = 2000000000 // 2 GB
  description = "Alarm threshold for the low-free-storage-space alarm"
}

variable "disk_burst_balance_too_low_threshold" {
  default     = 20
  description = "Alarm threshold for the low-disk-burst-balance alarm"
}

variable "memory_freeable_too_low_threshold" {
  default     = 256000000 // 256 MB
  description = "Alarm threshold for the low-freeable-memory alarm"
}

variable "memory_swap_usage_too_high_threshold" {
  default     = 256000000 // 256 MB
  description = "Alarm threshold for the high-swap-usage alarm"
}

variable "maximum_used_transaction_ids_too_high_threshold" {
  default     = 1000000000 // 1 billion. Half of total.
  description = "Alarm threshold for the high-maximum-transcation-ids alarm"
}

variable "read_iops_too_high_threshold" {
  default     = 3000 
  description = "Alarm threshold for the high-read-iops alarm"
}

variable "aws_notifications_slack_app_webhook_url" {
  type      = string
  sensitive = true
}

variable "create_rds_alarms_sns_and_slack" {
  default = true
}
