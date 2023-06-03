variable account_id {
  type = string
}

variable "primary_region" {
  type = string
}

variable "environment" {
  type = string
}

variable "aws_notifications_slack_app_webhook_url" {
  type = string
  sensitive = true
}